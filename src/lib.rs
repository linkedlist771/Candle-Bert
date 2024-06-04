#![allow(unused)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::path::PathBuf;
use tokenizers::{PaddingParams, Tokenizer};

fn build_model_and_tokenizer(
    model_dir_path: String,
    use_cuda: bool,
    use_pth: bool,
    approximate_gelu: bool,
) -> PyResult<(BertModel, Tokenizer)> {
    let device = if use_cuda {
        Device::cuda_if_available(0).map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?
    } else {
        Device::Cpu
    };
    println!("The model has been loaded on {:?}", device);
    let local_model_path = PathBuf::from(model_dir_path);
    let (config_filename, tokenizer_filename, weights_filename) = {
        let config = local_model_path.join("config.json");
        let tokenizer = local_model_path.join("tokenizer.json");
        let weights = if use_pth {
            local_model_path.join("pytorch_model.bin")
        } else {
            local_model_path.join("model.safetensors")
        };

        (config, tokenizer, weights)
    };

    let config =
        std::fs::read_to_string(config_filename).map_err(|e| PyIOError::new_err(e.to_string()))?;
    let mut config: Config =
        serde_json::from_str(&config).map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename)
        .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;

    let vb = if use_pth {
        VarBuilder::from_pth(&weights_filename, DTYPE, &device)
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?
    } else {
        unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)
                .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?
        }
    };
    if approximate_gelu {
        config.hidden_act = HiddenAct::GeluApproximate;
    }
    let model =
        BertModel::load(vb, &config).map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;
    Ok((model, tokenizer))
}

#[pyclass]
struct CandleBert {
    // #[pyo3(get, set)]
    model: BertModel,
    tokenizer: Tokenizer,
}

fn candle_core_error_to_pyerr(err: candle_core::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{}", err))
}

#[pymethods]
impl CandleBert {
    #[new]
    fn new(
        model_dir_path: String,
        use_cuda: bool,
        use_pth: bool,
        approximate_gelu: bool,
    ) -> PyResult<Self> {
        let (model, mut tokenizer) =
            build_model_and_tokenizer(model_dir_path, use_cuda, use_pth, approximate_gelu)?;

        Ok(CandleBert { model, tokenizer })
    }

    fn forward(&mut self, sentences: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        let device = &self.model.device;
        let n_sentences = sentences.len();

        if let Some(pp) = self.tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            self.tokenizer.with_padding(Some(pp));
        }

        let tokens = self
            .tokenizer
            .encode_batch(sentences, true)
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;

        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let token_ids = tokens.get_ids().to_vec();
                Ok(Tensor::new(token_ids.as_slice(), device)
                    .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?)
            })
            .collect::<PyResult<Vec<_>>>()?;

        let token_ids =
            Tensor::stack(&token_ids, 0).map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;

        let token_type_ids = token_ids
            .zeros_like()
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;

        let embeddings = self
            .model
            .forward(&token_ids, &token_type_ids)
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;
        //  use average to convert lenx7x384 => lenx384
        let (_n_sentence, n_tokens, _hidden_size) =
            embeddings.dims3().map_err(candle_core_error_to_pyerr)?;
        let summed_embeddings = embeddings.sum(1).map_err(candle_core_error_to_pyerr)?;
        let embeddings =
            (summed_embeddings / (n_tokens as f64)).map_err(candle_core_error_to_pyerr)?;

        let embeddings_vec = (0..n_sentences)
            .map(|i| {
                let embedding = embeddings
                    .get(i)
                    .map_err(candle_core_error_to_pyerr)?
                    .to_vec1()
                    .map_err(candle_core_error_to_pyerr)?;
                Ok(embedding)
            })
            .collect::<Result<Vec<_>, PyErr>>()?;

        Ok(embeddings_vec)
    }
}

/// This module is a python module implemented in Rust.
#[pymodule]
fn candle_bert(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CandleBert>()?;
    Ok(())
}

use crate::ml::model::FlappyNet;
use crate::MyAutodiffBackend;
use burn::backend::wgpu::WgpuDevice;
use burn::prelude::Module;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::AutodiffBackend;
use std::sync::OnceLock;
pub struct AMRessource<B: AutodiffBackend>(pub B::Device);
unsafe impl<B: AutodiffBackend> Sync for AMRessource<B> {}

static DEVICE: OnceLock<AMRessource<MyAutodiffBackend>> = OnceLock::new();

impl<B: AutodiffBackend> AMRessource<B> {
    pub fn weights_to_bytes(model: &FlappyNet<B>) -> Vec<u8> {
        BinBytesRecorder::<FullPrecisionSettings>::default()
            .record(model.clone().into_record(), ())
            .expect("failed to serialize weights")
    }
}

pub fn set_global_device(device: WgpuDevice) {
    DEVICE
        .set(AMRessource(device))
        .unwrap_or_else(|_| panic!("device already initialized"));
}

pub fn get_global_device() -> WgpuDevice {
    DEVICE
        .get()
        .expect("device not initialized — call init() from JS before constructing agents")
        .0
        .clone()
}

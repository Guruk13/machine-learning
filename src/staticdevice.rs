use crate::MyAutodiffBackend;
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::backend::AutodiffBackend;
use std::sync::OnceLock;

pub struct AMRessource<B: AutodiffBackend>(pub B::Device);
unsafe impl<B: AutodiffBackend> Sync for AMRessource<B> {}

static DEVICE: OnceLock<AMRessource<MyAutodiffBackend>> = OnceLock::new();

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

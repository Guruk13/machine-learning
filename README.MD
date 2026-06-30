build:


cargo build --release --target wasm32-unknown-unknown   --features "bevy/webgpu,bevy/bevy_winit,bevy/bevy_render,bevy/bevy_core_pipeline,bevy/bevy_sprite,bevy/bevy_text,bevy/bevy_ui,bevy/bevy_asset,bevy/png"
wasm-bindgen --out-dir ./out --target web   ./target/wasm32-unknown-unknown/release/flappy-bird.wasm

to run on web : 
python3 -c "
import http.server, socketserver

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

socketserver.TCPServer(('', 8080), Handler).serve_forever()
"

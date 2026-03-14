// Web Worker: loads WASM module, initializes model, runs inference.
// Runs off the main thread to keep the UI responsive.

/* eslint-disable @typescript-eslint/no-explicit-any */
let wasmModule: any = null;

self.onmessage = async (event: MessageEvent) => {
  const request = event.data;

  if (request.type === "init") {
    try {
      self.postMessage({
        type: "init:progress",
        stage: "wasm",
        message: "Loading WASM module...",
      });

      // Build absolute URLs — Workers don't resolve root-relative paths automatically
      const origin = self.location.origin;
      const wasmUrl = `${origin}/wasm/yolo26_rust_wasm_bg.wasm`;
      const glueUrl = `${origin}/wasm/yolo26_rust_wasm.js`;

      // Load the JS glue via dynamic import with variable to bypass TS module check
      const wasm = await (Function(
        "url",
        "return import(url)",
      )(glueUrl) as Promise<any>);
      // Fetch and compile WASM bytes, then init synchronously
      const wasmResp = await fetch(wasmUrl);
      const wasmBytes = await wasmResp.arrayBuffer();
      wasm.initSync({ module: wasmBytes });
      wasmModule = wasm;

      self.postMessage({
        type: "init:progress",
        stage: "model",
        message: "Initializing model...",
      });

      const weightsArray = new Uint8Array(request.weightsBuffer);
      wasm.init_model(weightsArray, request.modelSize);

      const sizeMB = parseFloat((weightsArray.length / 1e6).toFixed(1));
      self.postMessage({ type: "init:done", modelSizeMB: sizeMB });
    } catch (error) {
      self.postMessage({ type: "init:error", error: String(error) });
    }
  }

  if (request.type === "detect") {
    try {
      const resultJson: string = wasmModule.detect(
        request.pixels,
        request.width,
        request.height,
        request.confidenceThreshold,
      );
      self.postMessage({ type: "detect:done", resultJson });
    } catch (error) {
      self.postMessage({ type: "detect:error", error: String(error) });
    }
  }
};

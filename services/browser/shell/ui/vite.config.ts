import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
      "#repo": fileURLToPath(new URL("../../../../", import.meta.url))
    }
  },
  server: {
    host: true,
    port: 5174,
    fs: {
      allow: ["../../../../"]
    },
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      }
    }
  },
  preview: {
    host: true,
    port: 5174
  },
  build: {
    outDir: "dist",
    emptyOutDir: true
  },
});
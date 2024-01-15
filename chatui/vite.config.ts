/// <reference types='vitest' />
import { nxViteTsPaths } from '@nx/vite/plugins/nx-tsconfig-paths.plugin';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
	cacheDir: './node_modules/.vite/memgpt',

	server: {
		port: 4200,
		host: 'localhost',
		proxy: {
			'/api': {
				target: 'http://127.0.0.1:8283',
				changeOrigin: true,
			},
		},
	},

	preview: {
		port: 4300,
		host: 'localhost',
	},

	plugins: [react(), nxViteTsPaths()],

	// Uncomment this if you are using workers.
	// worker: {
	//  plugins: [ nxViteTsPaths() ],
	// },

	test: {
		globals: true,
		cache: {
			dir: './node_modules/.vitest',
		},
		environment: 'jsdom',
		include: ['src/**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}'],
	},
});

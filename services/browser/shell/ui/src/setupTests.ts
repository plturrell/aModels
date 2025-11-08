/// <reference types="vitest" />
import '@testing-library/jest-dom';
import { cleanup } from '@testing-library/react';
import * as matchers from '@testing-library/jest-dom/matchers';

// Extend Vitest's expect with jest-dom matchers
(globalThis as any).expect.extend(matchers);

// Cleanup after each test
(globalThis as any).afterEach(() => {
  cleanup();
});
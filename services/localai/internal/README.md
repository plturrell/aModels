# Internal Packages

This directory contains packages and code that are considered "internal" to the VaultGemma LocalAI application. 

## Purpose

In Go, any code within a directory named `internal` is only accessible by code in the same repository that shares a common ancestor directory. This means that other projects cannot import and use the packages defined here.

We use this convention to store code that is critical for the functioning of our server but is not intended to be a public, reusable library. This might include:

- Utility functions.
- Helper methods.
- Code that is tightly coupled to the specific implementation of our server.

By keeping this code here, we can freely refactor and modify it without worrying about breaking external applications that might depend on it.

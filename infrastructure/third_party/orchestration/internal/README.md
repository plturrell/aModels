# Internal Packages

This directory contains packages that are for internal use by the orchestration framework only.

## Purpose

In Go, any code within a directory named `internal` is only accessible by code within the same repository that shares a common ancestor directory. This is a compile-time restriction that prevents other projects from importing and depending on the packages defined here.

We use this convention to store code that is critical for the functioning of the framework but is not intended to be part of the public, stable API. This allows the project maintainers to freely refactor and modify this code without worrying about breaking external applications that might be using the framework.

## Potential Content

This directory might contain:

-   Implementations of interfaces that are not meant to be used directly.
-   Helper functions that are tightly coupled to the specific implementation of the framework's components.
-   Code that is experimental or subject to change.

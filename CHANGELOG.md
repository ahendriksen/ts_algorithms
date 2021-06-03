# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Support for pytorch 1.7, 1.8 and higher.
- Batched filtering in FDK and FBP reconstruction to conserve memory.
- Support for overwriting sinogram in FDK and FBP to conserve memory.
### Fixed
- FDK/FBP: Force filter to be same size as padded sinogram.
### Removed
- Support for pytorch versions below 1.7.
- `reject_acyclic_filter` flag: not necessary anymore now that complex
  multiplication is available.

## 0.1.0 - 2021-06-03
### Added
- Initial release.

[Unreleased]: https://www.github.com/ahendriksen/ts_algorithms/compare/v0.1.0...develop

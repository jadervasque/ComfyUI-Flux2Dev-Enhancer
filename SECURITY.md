# Security Policy

## Supported versions

Security fixes are applied to the latest published release and the default development branch. Older releases are not maintained unless explicitly stated in release notes.

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose user files, execute unintended code, bypass ComfyUI security controls, or compromise a host system.

Use GitHub's **Report a vulnerability** / private security advisory feature for this repository. Include:

- affected version or commit;
- ComfyUI version and installation method;
- operating system and Python version;
- reproduction steps or proof of concept;
- expected and observed impact;
- suggested mitigation, when known.

The maintainer will acknowledge a complete report as repository availability permits, assess scope, coordinate a fix, and publish disclosure details after affected users have a reasonable upgrade path.

## Scope

Security issues in this repository include unsafe file handling, unexpected code execution, injection through node inputs, unsafe serialization, dependency compromise, and model-loader interactions introduced by this extension.

Model files, ComfyUI itself, third-party custom nodes, quantized loaders, GPU drivers, and external services are separate projects. Reports may be redirected when the root cause is outside this repository.

## Responsible disclosure

Do not publish exploit details before a coordinated fix or mitigation is available. Good-faith research that follows this policy will be handled constructively.

<div align="center">

# QANTIS

**Quantum Autonomous Navigation, Tracking & Intelligence System**

*A quantum-native platform for next-generation autonomous systems.*

[![Edition](https://img.shields.io/badge/edition-community-2563eb?style=flat-square)](#editions)
[![Status](https://img.shields.io/badge/status-public--preview-10b981?style=flat-square)](#)
[![License](https://img.shields.io/badge/license-MIT-111827?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?style=flat-square)](#)
[![Built by](https://img.shields.io/badge/built%20by-Neura%20Parse%20Ltd-0f172a?style=flat-square)](#)

</div>

---

> ### Community Edition Notice
>
> You are looking at the **public, community-facing distribution** of QANTIS.
> This repository is intentionally a **basic edition** — it ships a clean, didactic surface of the platform so that researchers, students, and partners can understand the architecture and integrate at the API level.
>
> The **full QANTIS platform**, including production-grade modules, advanced optimisation backends, hardened mitigation pipelines, internal tooling, and the complete experimental harness, is maintained in a **private workspace reserved for Neura Parse collaborators and contracted partners**.
>
> Real benchmark runs, hardware campaign results, comparative studies, and superior performance figures obtained on production backends are **not published in this repository**. Selected outcomes are disclosed only through peer-reviewed publications, partner briefings, and formal engagements with Neura Parse Ltd.

---

## About

QANTIS is a **quantum-native framework** designed by Neura Parse Ltd for autonomous systems that must operate under uncertainty — drone swarms, GPS-denied navigation, multi-target surveillance, and intelligent decision making in adversarial environments.

The platform is structured as a **core framework** with a family of **purpose-built applications** running on top of it. Each application solves a real-world autonomy problem; the framework handles everything underneath — backends, mitigation, configuration, reproducibility.

---

## The Platform

<table>
  <tr>
    <td width="33%" valign="top">
      <h3>Quantum Common</h3>
      <em>The Framework</em>
      <p>The foundation layer of the QANTIS platform. Provides the unified backend abstraction, configuration system, error-mitigation pipeline, benchmarking and reproducibility infrastructure that every QANTIS application is built on.</p>
    </td>
    <td width="33%" valign="top">
      <h3>Quantum POMDP App</h3>
      <em>Decision Making Under Uncertainty</em>
      <p>An application focused on belief-state planning for autonomous agents operating with partial information. Targets navigation, mission planning and policy execution in environments where perception is incomplete or unreliable.</p>
    </td>
    <td width="33%" valign="top">
      <h3>Quantum MHT App</h3>
      <em>Multi-Target Tracking</em>
      <p>An application for large-scale multi-hypothesis tracking, designed for drone swarm surveillance and dense multi-target scenarios where classical association becomes intractable at scale.</p>
    </td>
  </tr>
</table>

> The full QANTIS platform extends well beyond autonomy. Additional applications — including **Quantum-Bio Intelligence** for computational life sciences, a dedicated **CRISPR** application for quantum-assisted genome design and target analysis, alongside modules covering sensor fusion, adversarial robustness, and mission-level orchestration — are developed inside the private collaborator workspace and are not part of this Community Edition.

---

## Editions

QANTIS is distributed in two editions. This repository corresponds to the first row.

| | **Community Edition** *(this repository)* | **Collaborator Edition** *(private)* |
|---|---|---|
| Audience | Public, academic, evaluation | Neura Parse partners & collaborators |
| Framework core | Included, basic surface | Full production framework |
| Applications | Reference subset (POMDP, MHT) | Complete suite — incl. Quantum-Bio Intelligence, CRISPR, and more |
| Backends | Standard public connectors | Hardened, optimised, multi-vendor |
| Error mitigation | Baseline pipeline | Full mitigation & calibration stack |
| Benchmarks & datasets | Illustrative only | Full experimental harness |
| Real hardware results | Not published here | Reserved for partners |
| Support | Community, best-effort | Dedicated engineering support |

Access to the Collaborator Edition is granted on a case-by-case basis through formal engagement with Neura Parse Ltd.

---

## Design Principles

- **Framework-first.** A single, well-defined core powers every QANTIS application. No application owns its own infrastructure.
- **Backend-agnostic.** Applications are written once and dispatched across simulators, gate-based hardware, and quantum annealers through a uniform interface.
- **Reproducibility by construction.** Every run is configuration-driven, versioned, and traceable.
- **Incremental adoption.** Heavy quantum SDKs are optional; the framework installs cleanly with a minimal scientific Python stack.
- **Hybrid by default.** Quantum routines are designed to interoperate with classical baselines, not to replace them.

---

## Getting Started

```bash
git clone https://github.com/neuraparse/qantis.git
cd qantis
uv sync
```

Optional quantum backends and visualisation extras are available as installable groups. See the in-package documentation for the relevant extras for your environment.

> The Community Edition is intended for **architectural exploration, integration testing, and educational use**. It is not a substitute for the Collaborator Edition in production or research-grade evaluation contexts.

---

## What Is *Not* Included

To set expectations clearly, the following are deliberately **outside the scope** of this Community Edition:

- Production-grade application variants and internal QANTIS modules
- Calibrated, hardware-tuned mitigation and optimisation pipelines
- Real measurement data, raw hardware traces, and campaign artefacts
- Comparative benchmarks and performance figures vs. classical state-of-the-art
- Confidential datasets, scenarios, and mission profiles developed with partners
- Any results that constitute commercial or research advantage for Neura Parse Ltd

These remain proprietary to the Collaborator Edition. Public communication of selected results occurs through formal channels only.

---

## Roadmap (Public Track)

The Community Edition follows a separate, slower release track than the internal platform. Public milestones focus on:

- Stabilising the framework API surface
- Expanding didactic examples and integration guides
- Publishing reference notebooks aligned with peer-reviewed outputs
- Hardening installation and onboarding for new contributors

Items on the internal roadmap — including production applications, advanced backends, and benchmark releases — are governed independently and are not announced here.

---

## Contributing

Community contributions are welcome for documentation, examples, integration fixes, and onboarding improvements. Contributions affecting the framework core or any QANTIS application are reviewed against the internal platform and may be deferred or adapted to maintain compatibility with the Collaborator Edition.

For substantial collaboration, joint research, or evaluation under the Collaborator Edition, please contact Neura Parse Ltd directly.

---

## Citation

If you reference QANTIS in academic or technical work, please cite the official Neura Parse publications associated with the platform. Citing this repository alone is not sufficient to attribute results obtained on the full platform.

---

## License

Released under the **MIT License**. See [LICENSE](LICENSE) for details.

The MIT license applies to the source code in this Community Edition repository. It does not extend to internal QANTIS modules, datasets, calibration data, or experimental results maintained outside this repository.

---

<div align="center">

**QANTIS** — engineered by **Neura Parse Ltd**.

*Quantum technology for autonomous systems that have to work in the real world.*

</div>

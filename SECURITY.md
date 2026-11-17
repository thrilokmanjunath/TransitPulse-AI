# Security Policy — TransitPulse-AI

## Supported Versions

| Version | Supported          | Security Maintenance Status |
| ------- | ------------------ | --------------------------- |
| 1.0.x   | :white_check_mark: | Active production support   |

---

## Reporting a Vulnerability

The TransitPulse-AI project takes the security of urban mobility data pipelines and machine learning inference endpoints seriously.

If you discover a security vulnerability, please **DO NOT** create a public GitHub issue. Instead, report it privately to:

**Security Lead:** Thrilok Manjunath  
**Email:** `thrilokmanjunath@gmail.com`  
**Response Window:** Within 48 hours with an initial acknowledgment and mitigation timeline.

---

## Security Invariants & Hardening Measures

1. **Zero Secret Storage:**
   - No hardcoded API keys, private tokens, or database passwords exist in source files or version control history.
   - All runtime variables are configured via environment variables and `.env.example`.
2. **Static Code Analysis:**
   - Codebase scanned continuously via Bandit with zero High or Medium severity vulnerabilities permitted.
3. **Dependency Integrity:**
   - Python dependencies audited against the Advisory Database using `pip-audit`.
4. **Input Sanitization & Injection Defense:**
   - Strict Pydantic typing and validation on all REST endpoints (`POST /api/v1/predict`, `POST /api/v1/scenario`).
   - Range validation prevents buffer overruns and floating-point injection attacks.
5. **Non-Root Execution:**
   - Docker containers run isolated container workloads with minimal privileges and drop unnecessary capabilities.

## Memory Bank Protocol (Required)

This project uses a Memory Bank system in `memory-bank/` for cross-session context continuity.

### At Session Start — ALWAYS:
1. Read `memory-bank/RULES.md` — all rules are there
2. Read `memory-bank/activeContext.md` — current work and decisions
3. Read `memory-bank/progress.md` — current status
4. Read other files as needed (`systemPatterns.md`, `techContext.md`, `productContext.md`, `projectbrief.md`)

### During Work — Update When:
- Feature completed → update `activeContext.md` + `progress.md`
- Architecture decision made → update `systemPatterns.md`
- New dependency added → update `techContext.md`
- User preference learned → update `activeContext.md`

### Special Commands:
- `memory bank update` / `memory bank güncelle` → Review and update ALL memory bank files
- `memory bank status` / `memory bank durumu` → Show current status summary
- `memory bank read` / `memory bank oku` → Read all files and present context

### NEVER:
- Modify `memory-bank/RULES.md` (it's immutable)
- Write secrets (API keys, tokens, passwords) to memory bank files
- Skip reading memory bank at session start

---

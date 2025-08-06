#!/usr/bin/env zsh
# Robust acceptance runner for zsh. Never exits early; prints a SUMMARY at the end.

set -u  # no -e and no pipefail; we handle exit codes ourselves

# ---------- utils ----------
have() { command -v "$1" >/dev/null 2>&1 }
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { print -P "%F{cyan}[$(ts)]%f $*"; }
warn() { print -P "%F{yellow}[$(ts)] WARN:%f $*"; }

OUT="acceptance_artifacts"
mkdir -p "$OUT"

# ---------- JSON helpers ----------
sanitize_json() {
  # sanitize_json <infile> <outfile>
  local infile="$1" outfile="$2"
  python3 - "$infile" "$outfile" <<'PY' 2>/dev/null
import sys, re, json
def strip_ansi(s: str) -> str:
    return re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', s)
def drop_ctrl(s: str) -> str:
    return ''.join(ch for ch in s if ch in '\n\r\t' or ord(ch) >= 32)
def extract_first_json(s: str) -> str | None:
    for opener, closer in (('{','}'), ('[',']')):
        start = s.find(opener)
        while start != -1:
            i = start
            depth = 0
            in_str = False
            esc = False
            while i < len(s):
                ch = s[i]
                if in_str:
                    if esc: esc = False
                    elif ch == '\\': esc = True
                    elif ch == '"': in_str = False
                else:
                    if ch == '"': in_str = True
                    elif ch == opener: depth += 1
                    elif ch == closer:
                        depth -= 1
                        if depth == 0:
                            candidate = s[start:i+1]
                            try:
                                obj = json.loads(candidate)
                                return json.dumps(obj, ensure_ascii=False, indent=2)
                            except Exception:
                                break
                i += 1
            start = s.find(opener, start+1)
    return None

inp, out = sys.argv[1], sys.argv[2]
raw = open(inp, 'rb').read().decode('utf-8', errors='replace')
clean = drop_ctrl(strip_ansi(raw))
try:
    obj = json.loads(clean)
    data = json.dumps(obj, ensure_ascii=False, indent=2)
except Exception:
    data = extract_first_json(clean)
if not data:
    sys.exit(2)
open(out, 'w', encoding='utf-8').write(data)
PY
}

json_ok() {
  # json_ok <file>
  if have jq && jq -e '.' "$1" >/dev/null 2>&1; then
    return 0
  fi
  local cleaned="${1%.json}.clean.json"
  sanitize_json "$1" "$cleaned" || return 1
  mv -f "$cleaned" "$1"
  have jq && jq -e '.' "$1" >/dev/null 2>&1
}

safe_jq() {
  local f="$1"; shift
  if have jq; then jq -r "$*" "$f" 2>/dev/null; else print -- "n/a"; return 1; fi
}

count_ticks() { local s="$1" f="$2"; [[ -f "$f" ]] && grep -o "\`$s\`" "$f" | wc -l | tr -d ' ' || echo 0 }

# (опционально) меньше цветов от CLI
export NO_COLOR=1

# ---------- choose CLI (as an ARRAY) ----------
typeset -a cli
if have uv; then
  if uv run prompt-distil --help >/dev/null 2>&1; then
    cli=(uv run prompt-distil)
  else
    cli=(uv run python -m prompt_distil.cli)
  fi
elif [[ -x ".venv/bin/python" ]]; then
  cli=(.venv/bin/python -m prompt_distil.cli)
else
  cli=(python -m prompt_distil.cli)
fi

# ---------- Preflight ----------
log "Preflight"
if have uv; then uv sync >/dev/null 2>&1 || true; fi
if have uv; then uv run python -m pytest -q | tee "$OUT/pytest.txt"; else python -m pytest -q | tee "$OUT/pytest.txt"; fi

# ---------- Build symbol index ----------
log "Build symbol index"
"${cli[@]}" index --project-root . >/dev/null 2>"$OUT/index.err" || true

# ---------- 1) Rules EN ----------
log "1) Rules EN"
"${cli[@]}" distill --project-root . \
  --text "rewrite delete_task test for 404 and add logging in login handler" \
  --lex-mode rules --profile std --format markdown \
  > "$OUT/rules_en.md" 2>"$OUT/rules_en.err" || true
DEL_COUNT=$(count_ticks "delete_task" "$OUT/rules_en.md")
LOGIN_COUNT=$(count_ticks "login_handler" "$OUT/rules_en.md")

# ---------- 2) Rules RU (JSON) ----------
log "2) Rules RU"
"${cli[@]}" distill --project-root . \
  --text "переписать тест на удаление задачи; добавить логирование логина" \
  --lex-mode rules --profile std --format json \
  > "$OUT/rules_ru.json" 2>"$OUT/rules_ru.err" || true
if json_ok "$OUT/rules_ru.json"; then
  RU_KEYS=$(safe_jq "$OUT/rules_ru.json" 'keys')
  RU_GOAL=$(safe_jq "$OUT/rules_ru.json" '.goal // "n/a"')
  RU_ENTS=$(safe_jq "$OUT/rules_ru.json" '[.known_entities[]? | "\(.path // ""):\(.symbol // "")"] | join(", ") // "n/a"')
else
  RU_ERR=$(head -n 2 "$OUT/rules_ru.err" 2>/dev/null | tr -d '\r')
fi

# ---------- 3) Hybrid FR (JSON) ----------
log "3) Hybrid FR"
"${cli[@]}" distill --project-root . \
  --text "Réécris le test de suppression de tâche et ajoute le logging de connexion" \
  --lex-mode hybrid --profile std --format json \
  > "$OUT/hybrid_fr.json" 2>"$OUT/hybrid_fr.err" || true
if json_ok "$OUT/hybrid_fr.json"; then
  FR_KEYS=$(safe_jq "$OUT/hybrid_fr.json" 'keys')
  FR_GOAL=$(safe_jq "$OUT/hybrid_fr.json" '.goal // "n/a"')
  FR_ENTS=$(safe_jq "$OUT/hybrid_fr.json" '[.known_entities[]? | "\(.path // ""):\(.symbol // "")"] | join(", ") // "n/a"')
else
  FR_ERR=$(head -n 2 "$OUT/hybrid_fr.err" 2>/dev/null | tr -d '\r')
fi

# ---------- 4) Hybrid EN (JSON) ----------
log "4) Hybrid EN"
"${cli[@]}" distill --project-root . \
  --text "please improve the sign-in handler logs and rewrite the test that deletes a task" \
  --lex-mode hybrid --profile std --format json \
  > "$OUT/hybrid_en.json" 2>"$OUT/hybrid_en.err" || true
if json_ok "$OUT/hybrid_en.json"; then
  EN_KEYS=$(safe_jq "$OUT/hybrid_en.json" 'keys')
  EN_GOAL=$(safe_jq "$OUT/hybrid_en.json" '.goal // "n/a"')
  EN_ENTS=$(safe_jq "$OUT/hybrid_en.json" '[.known_entities[]? | "\(.path // ""):\(.symbol // "")"] | join(", ") // "n/a"')
else
  EN_ERR=$(head -n 2 "$OUT/hybrid_en.err" 2>/dev/null | tr -d '\r')
fi

# ---------- 5) File input RU (Markdown) ----------
log "5) File input RU"
print -- "переписать тест на удаление задачи; добавить логирование логина" > "$OUT/ru.txt"
"${cli[@]}" distill --project-root . --file "$OUT/ru.txt" \
  --profile std --format markdown \
  > "$OUT/file_ru.md" 2>"$OUT/file_ru.err" || true
FILE_RU_TICKS=$(count_ticks "delete_task" "$OUT/file_ru.md")

# ---------- 6) ASR pipeline (optional) ----------
log "6) ASR pipeline (optional)"
if have say && have uv; then
  say -v Milena "переписать тест на удаление задачи и добавить логирование логина" -o "$OUT/ru.aiff" 2>/dev/null || true
  afconvert -f WAVE -d LEI16 "$OUT/ru.aiff" "$OUT/ru.wav" 2>/dev/null || true
  "${cli[@]}" from-audio "$OUT/ru.wav" --profile short --format json > "$OUT/asr_ru.json" 2>"$OUT/asr_ru.err" || true
  ASR_HEAD=$(head -n 40 "$OUT/asr_ru.json" 2>/dev/null)
else
  ASR_HEAD="(skipped)"
fi

# ---------- 7) Error handling ----------
log "7) Error handling without OPENAI_API_KEY"
NO_KEY_RC=0
if [[ -z "${SKIP_NO_KEY:-}" ]]; then
  prev_key="${OPENAI_API_KEY:-}"; unset OPENAI_API_KEY
  "${cli[@]}" distill --project-root . --text "any" --format markdown > "$OUT/no_key.txt" 2>&1
  NO_KEY_RC=$?
  [[ -n "$prev_key" ]] && export OPENAI_API_KEY="$prev_key"
else
  print -- "(skipped) set SKIP_NO_KEY=1" > "$OUT/no_key.txt"
fi

# ---------- 8) Help & env ----------
log "8) CLI help & env snapshot"
"${cli[@]}" --help > "$OUT/cli_help.txt" 2>&1 || true
if have uv; then uv run python -m pip list > "$OUT/pip_list.txt" 2>&1; else python -m pip list > "$OUT/pip_list.txt" 2>&1; fi

# ---------- SUMMARY ----------
print
print "==================== SUMMARY (paste this) ===================="

print "Pytest:"
passed="$(grep -Eo '[0-9]+ passed' "$OUT/pytest.txt" | head -n1)"
print "  ${passed:-see acceptance_artifacts/pytest.txt}"
print

print "[Rules EN] Backticks in rules_en.md (counts):"
print "  delete_task: $DEL_COUNT"
print "  login_handler: $LOGIN_COUNT"
print

print "[Rules RU] JSON overview:"
if json_ok "$OUT/rules_ru.json"; then
  print "  root_keys: ${RU_KEYS:-[]}"
  print "  goal: ${RU_GOAL:-n/a}"
  print "  known_entities: ${RU_ENTS:-n/a}"
else
  print "  !! JSON parse error: ${RU_ERR:-see acceptance_artifacts/rules_ru.err}"
fi
print

print "[Hybrid FR] JSON overview (EXPECTED neutral anchors -> auto lexicon):"
if json_ok "$OUT/hybrid_fr.json"; then
  print "  root_keys: ${FR_KEYS:-[]}"
  print "  goal: ${FR_GOAL:-n/a}"
  print "  known_entities: ${FR_ENTS:-n/a}"
else
  print "  !! JSON parse error: ${FR_ERR:-see acceptance_artifacts/hybrid_fr.err}"
fi
print

print "[Hybrid EN] JSON overview:"
if json_ok "$OUT/hybrid_en.json"; then
  print "  root_keys: ${EN_KEYS:-[]}"
  print "  goal: ${EN_GOAL:-n/a}"
  print "  known_entities: ${EN_ENTS:-n/a}"
else
  print "  !! JSON parse error: ${EN_ERR:-see acceptance_artifacts/hybrid_en.err}"
fi
print

print "[File RU] Backtick for delete_task in file_ru.md (count):"
print "  ${FILE_RU_TICKS:-0}"
print

print "[ASR] If ran, first lines of asr_ru.json:"
print -- "${ASR_HEAD:-}"
print

print "[No key] Exit code and first lines:"
print "  exit=${NO_KEY_RC:-}"
head -n 2 "$OUT/no_key.txt" 2>/dev/null | sed 's/^/  /'
print "=============================================================="

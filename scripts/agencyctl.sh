#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Ajans sistemi kontrol araci
#
#   ./scripts/agencyctl.sh start [--cycles N] [--sleep S] [--backend X ...]
#   ./scripts/agencyctl.sh stop
#   ./scripts/agencyctl.sh restart
#   ./scripts/agencyctl.sh status     # surec + DB paneli
#   ./scripts/agencyctl.sh logs       # canli log (tail -f)
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDFILE="$ROOT/runtime/daemon.pid"
LOGFILE="$ROOT/runtime/logs/daemon.out"
cd "$ROOT"

mkdir -p "$ROOT/runtime/logs"

is_running() {
  [[ -f "$PIDFILE" ]] || return 1
  local pid
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  [[ -n "${pid:-}" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

start() {
  if is_running; then
    echo "Ajans sistemi zaten calisiyor (PID=$(cat "$PIDFILE"))."
    return 0
  fi
  nohup python3 main.py run "$@" >>"$LOGFILE" 2>&1 &
  echo $! >"$PIDFILE"
  sleep 2
  if is_running; then
    echo "Ajans sistemi baslatildi (PID=$(cat "$PIDFILE"))."
    echo "Log: $LOGFILE"
    echo "Izleme: ./scripts/agencyctl.sh logs | Durum: ./scripts/agencyctl.sh status"
  else
    echo "Baslatma basarisiz. Log: $LOGFILE" >&2
    tail -20 "$LOGFILE" >&2 || true
    return 1
  fi
}

stop() {
  if ! is_running; then
    echo "Ajans sistemi calismiyor."
    rm -f "$PIDFILE"
    return 0
  fi
  local pid
  pid="$(cat "$PIDFILE")"
  echo "Durduruluyor (PID=$pid). Mevcut dongu tamamlaninca cikacak..."
  kill -TERM "$pid" 2>/dev/null || true
  # Graceful cikis: mevcut dongu + state yazimi icin en fazla 45 sn
  for _ in $(seq 1 45); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PIDFILE"
      echo "Durduruldu (state ve raporlar yazildi)."
      return 0
    fi
    sleep 1
  done
  echo "Zaman asimi; zorla kapatiliyor (SIGKILL)." >&2
  kill -KILL "$pid" 2>/dev/null || true
  rm -f "$PIDFILE"
}

status() {
  if is_running; then
    echo "Durum: CALISIYOR (PID=$(cat "$PIDFILE"))"
    ps -o pid,etime,%cpu,%mem,cmd -p "$(cat "$PIDFILE")" || true
  else
    echo "Durum: DURDU (ajanlar idle)"
  fi
  echo
  python3 main.py status --limit 5
}

case "${1:-status}" in
  start)   shift; start "$@" ;;
  stop)    stop ;;
  restart) stop; shift || true; start "$@" ;;
  status)  status ;;
  logs)    tail -f "$LOGFILE" ;;
  *)
    echo "Kullanim: $0 {start [main.py run secenekleri]|stop|restart|status|logs}" >&2
    exit 2
    ;;
esac

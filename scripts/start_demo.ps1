<#
  연구실 지능형 CCTV 데모 한 번에 실행 (Windows)

  순서: Ollama 확인 -> 백엔드(uvicorn) 실행 -> AI(main.py) 실행 -> 브라우저 열기
  이 창을 Ctrl+C 하거나 AI가 끝나면, 이 스크립트가 띄운 프로세스만 같이 정리한다.

  사용법 (conda 환경을 먼저 켜고):
    conda activate lab-cctv
    powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1
  옵션:
    -Port 8000      백엔드 포트
    -NoBrowser      브라우저를 열지 않음 (SSH 접속처럼 화면이 없을 때)
    -NoVlm          AI를 규칙만으로 실행 (python -m src.main --no-vlm)
#>
[CmdletBinding()]
param(
    [int]$Port = 8000,
    [switch]$NoBrowser,
    [switch]$NoVlm
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BaseUrl = "http://127.0.0.1:$Port"

# 이 스크립트가 직접 띄운 프로세스만 담는다. (원래 켜져 있던 건 건드리지 않음)
$Script:Started = @()

function Write-Step([string]$text) { Write-Host "`n[단계] $text" -ForegroundColor Cyan }
function Write-Ok([string]$text)   { Write-Host "  OK  $text" -ForegroundColor Green }
function Write-Info([string]$text) { Write-Host "      $text" -ForegroundColor DarkGray }

function Stop-Started {
    if ($Script:Started.Count -eq 0) { return }
    Write-Host "`n[정리] 이 스크립트가 띄운 프로세스를 종료합니다." -ForegroundColor Yellow
    foreach ($item in $Script:Started) {
        $p = $item.Process
        if ($null -eq $p) { continue }
        try {
            $p.Refresh()
            if (-not $p.HasExited) {
                Stop-Process -Id $p.Id -Force -ErrorAction Stop
                Write-Info "$($item.Name) 종료 (PID $($p.Id))"
            }
        } catch {
            Write-Info "$($item.Name) 종료 실패 - 작업 관리자에서 직접 닫아주세요 (PID $($p.Id))"
        }
    }
    $Script:Started = @()
}

function Stop-WithError([string]$where, [string]$why, [string]$howToFix) {
    Write-Host "`n[실패] $where 에서 막혔습니다." -ForegroundColor Red
    Write-Host "  원인: $why" -ForegroundColor Red
    if ($howToFix) { Write-Host "  해결: $howToFix" -ForegroundColor Yellow }
    Stop-Started
    exit 1
}

function Test-Url([string]$url, [int]$TimeoutSec = 3) {
    try {
        Invoke-RestMethod -Uri $url -TimeoutSec $TimeoutSec -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Wait-Url([string]$url, [int]$Seconds, [System.Diagnostics.Process]$Watch) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        if ($null -ne $Watch) {
            $Watch.Refresh()
            if ($Watch.HasExited) { return $false }   # 프로세스가 먼저 죽으면 더 기다리지 않음
        }
        if (Test-Url $url 2) { return $true }
        Start-Sleep -Milliseconds 700
    }
    return $false
}

try {
    Set-Location $ProjectRoot
    Write-Host "연구실 지능형 CCTV 데모 시작" -ForegroundColor White
    Write-Info "작업 폴더: $ProjectRoot"

    # ---------- 0) 파이썬 환경 ----------
    Write-Step "파이썬 환경 확인"
    $py = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $py) {
        Stop-WithError "파이썬 확인" "python 명령을 찾을 수 없습니다." "conda activate lab-cctv 를 먼저 실행한 뒤 다시 시도하세요."
    }
    if ($env:CONDA_DEFAULT_ENV -and $env:CONDA_DEFAULT_ENV -ne "lab-cctv") {
        Write-Info "주의: 현재 conda 환경이 '$($env:CONDA_DEFAULT_ENV)' 입니다 (예상: lab-cctv)"
    }
    python -c "import fastapi, uvicorn, cv2, ultralytics" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Stop-WithError "파이썬 환경 확인" "필요한 패키지(fastapi/uvicorn/opencv/ultralytics)를 불러오지 못했습니다." "conda activate lab-cctv 후 python -m pip install -r requirements.txt 를 실행하세요."
    }
    Write-Ok "python 과 필요한 패키지 확인됨"

    # ---------- 1) Ollama ----------
    Write-Step "Ollama 확인 (LLM 요약/질문용)"
    $ollamaUrl = if ($env:OLLAMA_URL) { $env:OLLAMA_URL } else { "http://localhost:11434" }
    if (Test-Url "$ollamaUrl/api/tags") {
        Write-Ok "Ollama 가 이미 켜져 있습니다 ($ollamaUrl)"
    } else {
        $ollama = Get-Command ollama -ErrorAction SilentlyContinue
        if ($null -eq $ollama) {
            Stop-WithError "Ollama 확인" "$ollamaUrl 에 연결되지 않고, ollama 명령도 없습니다." "https://ollama.com 에서 Ollama 를 설치하거나, 이미 설치했다면 'ollama serve' 를 먼저 실행하세요."
        }
        Write-Info "Ollama 가 꺼져 있어 'ollama serve' 로 켭니다."
        $proc = Start-Process -FilePath $ollama.Source -ArgumentList "serve" -PassThru -WindowStyle Hidden
        $Script:Started += [pscustomobject]@{ Name = "Ollama"; Process = $proc }
        if (-not (Wait-Url "$ollamaUrl/api/tags" 20 $proc)) {
            Stop-WithError "Ollama 시작" "20초 안에 $ollamaUrl 가 응답하지 않았습니다." "터미널을 따로 열어 'ollama serve' 를 실행하고 오류 메시지를 확인하세요."
        }
        Write-Ok "Ollama 시작됨"
    }

    # 모델이 받아져 있는지 확인 (없으면 멈추지 않고 알려만 줌)
    try {
        $tags = Invoke-RestMethod -Uri "$ollamaUrl/api/tags" -TimeoutSec 5
        $have = @($tags.models | ForEach-Object { $_.name })
        foreach ($need in @("exaone3.5:7.8b", "qwen3-vl:4b-instruct")) {
            if (-not ($have -contains $need)) {
                Write-Info "주의: 모델 '$need' 이(가) 없습니다. 필요하면 'ollama pull $need' 를 실행하세요."
            }
        }
    } catch { }

    # ---------- 2) 백엔드 ----------
    Write-Step "백엔드(uvicorn) 실행"
    if (Test-Url "$BaseUrl/health") {
        Write-Ok "백엔드가 이미 $BaseUrl 에서 돌고 있습니다 (그대로 사용)"
    } else {
        $proc = Start-Process -FilePath "python" `
            -ArgumentList "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "$Port" `
            -WorkingDirectory $ProjectRoot -PassThru
        $Script:Started += [pscustomobject]@{ Name = "백엔드(uvicorn)"; Process = $proc }
        if (-not (Wait-Url "$BaseUrl/health" 40 $proc)) {
            Stop-WithError "백엔드 실행" "40초 안에 $BaseUrl/health 가 응답하지 않았습니다 (포트 $Port 가 이미 쓰이는 중이거나 서버가 바로 종료됨)." "백엔드 창의 오류 메시지를 확인하거나, -Port 9000 처럼 다른 포트로 다시 실행하세요."
        }
        Write-Ok "백엔드 준비됨 ($BaseUrl)"
    }

    # ---------- 3) AI ----------
    Write-Step "AI(main.py) 실행"
    $aiArgs = @("-m", "src.main")
    if ($NoVlm) { $aiArgs += "--no-vlm" }
    $ai = Start-Process -FilePath "python" -ArgumentList $aiArgs -WorkingDirectory $ProjectRoot -PassThru -NoNewWindow
    $Script:Started += [pscustomobject]@{ Name = "AI(main.py)"; Process = $ai }
    Start-Sleep -Seconds 5
    $ai.Refresh()
    if ($ai.HasExited) {
        Stop-WithError "AI 실행" "main.py 가 시작하자마자 종료됐습니다 (종료 코드 $($ai.ExitCode))." "카메라가 연결되어 있는지, .env 의 CAMERA_INDEX 값이 맞는지 확인하세요. 화면 창을 띄울 수 없는 환경(SSH 등)에서도 종료될 수 있습니다."
    }
    Write-Ok "AI 실행 중 (PID $($ai.Id))"

    # ---------- 4) 대시보드 ----------
    Write-Step "대시보드 열기"
    if ($NoBrowser) {
        Write-Info "-NoBrowser 옵션이라 열지 않습니다. 주소: $BaseUrl/"
    } else {
        try {
            Start-Process $BaseUrl
            Write-Ok "브라우저에서 $BaseUrl/ 를 열었습니다"
        } catch {
            Write-Info "브라우저를 열지 못했습니다. 직접 $BaseUrl/ 로 접속하세요."
        }
    }

    Write-Host "`n준비 끝. 이 창에서 Ctrl+C 를 누르면 전부 종료됩니다." -ForegroundColor White
    Wait-Process -Id $ai.Id
    Write-Host "`nAI(main.py)가 종료됐습니다." -ForegroundColor White
}
finally {
    Stop-Started
}

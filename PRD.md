# PRD: YOLO26 Rust WASM HTML Demo

---

## 1. Executive Summary

### Problem Statement

브라우저에서 YOLO 기반 객체 탐지를 수행하려면 보통 ONNX Runtime + JavaScript 조합을 사용하는데, 이 경우 전처리/후처리 로직이 읽기 쉬운 JS로 노출된다. 모델 추론 파이프라인을 보호하면서 브라우저에서 동작하는 방법이 필요하다.

### Proposed Solution

YOLO26n 모델을 **Rust → WASM**으로 컴파일하여, 전처리·추론·후처리 전체 파이프라인을 **불투명한 WASM 바이너리** 안에 캡슐화한다. 단일 `index.html`에서 이미지를 업로드하면 WASM 모듈이 추론을 수행하고, Canvas 위에 bounding box를 렌더링하는 **오픈소스 데모 프로젝트**.

### Success Criteria

| 기준 | 측정 방법 |
|------|----------|
| `wasm-pack build`로 WASM 바이너리 정상 빌드 | CI에서 빌드 성공 |
| 브라우저에서 WASM 모듈 + 모델 로드 완료 | Chrome, Mobile Chrome에서 에러 없이 로드 |
| 이미지 업로드 시 YOLO26n 추론 실행 및 결과 반환 | COCO val 이미지 5장으로 검증, bbox 좌표가 올바른 위치 |
| Canvas 위에 bounding box + class + confidence 시각화 | 육안 확인 + 좌표 값 검증 |
| 정적 파일 서빙만으로 동작 (서버 사이드 추론 없음) | `npx serve .`로 실행 후 전체 기능 동작 |
| 모바일 브라우저 동작 | Android Chrome + iOS Safari에서 이미지 업로드 → 추론 → 결과 표시 |

---

## 2. User Experience & Functionality

### User Personas

| 페르소나 | 설명 | 동기 |
|---------|------|------|
| **웹 개발자** | YOLO를 브라우저에서 돌리고 싶은 프론트엔드/풀스택 개발자 | JS 없이 WASM만으로 추론 파이프라인을 구현하는 레퍼런스가 필요 |
| **ML 엔지니어** | 모델을 엣지(브라우저)에 배포하려는 연구자/엔지니어 | Rust + WASM 기반 모델 서빙 방법을 검증하고 싶음 |
| **프라이버시 도구 개발자** | 서버 전송 없이 로컬에서 이미지 처리하는 도구를 만들려는 개발자 | dontsendfile.com 같은 서비스의 추론 모듈로 통합하고 싶음 |

### User Stories

**US-1: 이미지 업로드 및 추론**
> As a 개발자, I want to 브라우저에서 이미지를 업로드하고 YOLO26 추론 결과를 즉시 확인 so that WASM 기반 추론이 실제로 동작하는지 검증할 수 있다.

Acceptance Criteria:
- 이미지 선택 버튼 또는 드래그앤드롭으로 JPEG/PNG 이미지를 업로드할 수 있다
- 업로드 즉시 WASM 모듈이 추론을 실행한다
- 추론 중 "Detecting..." 상태가 표시된다
- 추론 완료 후 탐지 수와 추론 시간(ms)이 표시된다

**US-2: 결과 시각화**
> As a 개발자, I want to 탐지 결과를 bounding box로 시각화 so that 모델이 올바르게 동작하는지 직관적으로 확인할 수 있다.

Acceptance Criteria:
- Canvas 위에 원본 이미지가 렌더링된다
- 각 탐지에 대해 bounding box가 올바른 위치에 오버레이된다
- 각 box 상단에 class name과 confidence(%) 라벨이 표시된다
- 클래스별로 고유한 색상이 적용된다
- 탐지 목록이 텍스트로도 표시된다 (class, confidence, 좌표)

**US-3: Confidence 조절**
> As a 개발자, I want to confidence threshold를 슬라이더로 조절 so that 다양한 임계값에서의 탐지 결과를 비교할 수 있다.

Acceptance Criteria:
- 슬라이더 범위: 0.05 ~ 1.0, 기본값 0.25
- 슬라이더 변경 시 모델 재추론 없이 후처리 결과만 필터링하여 Canvas를 즉시 업데이트한다
- 현재 threshold 값이 슬라이더 옆에 표시된다

**US-4: 모바일 사용**
> As a 모바일 사용자, I want to 스마트폰에서도 이미지를 선택하고 결과를 확인 so that PC 없이도 데모를 체험할 수 있다.

Acceptance Criteria:
- 모바일 브라우저에서 페이지가 반응형으로 렌더링된다
- 카메라 촬영 또는 갤러리에서 이미지를 선택할 수 있다 (`accept="image/*"`)
- Canvas와 결과 목록이 화면 너비에 맞게 조정된다

**US-5: 모델 가중치 보호 (v2 — 최종 배포 시)**
> As a 서비스 운영자, I want to 모델 가중치를 WASM 바이너리에 임베딩 so that 사용자가 모델 파일을 별도로 다운로드/추출하기 어렵게 한다.

Acceptance Criteria:
- `include_bytes!()` 매크로로 SafeTensors 파일이 WASM 바이너리에 컴파일 타임에 포함된다
- 별도 모델 파일 fetch 없이 WASM 로드만으로 추론 준비가 완료된다
- 네트워크 탭에 모델 파일 다운로드가 노출되지 않는다

### Non-Goals (이 프로젝트에서 구현하지 않는 것)

- Blur/익명화/모자이크 기능 (dontsendfile.com 통합 시 별도 구현)
- Box 편집 UI (이동, 리사이즈, 삭제, 추가)
- React/Next.js/프레임워크 의존
- 멀티스레딩 (SharedArrayBuffer, COOP/COEP 헤더 불필요)
- WebGPU 가속
- 내보내기(Export) 기능
- 비디오 추론
- 커스텀 모델 학습

---

## 3. Technical Specifications

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser                                                         │
│                                                                  │
│  ┌─ index.html (Vanilla JS) ─────────────────────────────────┐  │
│  │                                                            │  │
│  │  1. <input type="file"> → 이미지 선택                      │  │
│  │  2. Canvas API → 이미지 디코딩 → RGBA pixels               │  │
│  │  3. WASM 호출: detect(pixels, w, h, threshold)            │  │
│  │  4. JSON 결과 파싱 → Canvas 2D bounding box 렌더링         │  │
│  │                                                            │  │
│  └────────────┬───────────────────────────┬───────────────────┘  │
│               │ RGBA pixels               │ JSON results         │
│               ▼                           │                      │
│  ┌─ Rust WASM Module ────────────────────┐│                      │
│  │                                        ││                      │
│  │  preprocess.rs                         ││                      │
│  │   └ RGBA→RGB, letterbox 640x640,      ││                      │
│  │     normalize [0,1], HWC→CHW          ││                      │
│  │                                        ││                      │
│  │  model.rs                              ││                      │
│  │   └ YOLO26n forward pass              ││                      │
│  │     (candle / onnx-runtime / tract)   ││                      │
│  │                                        ││                      │
│  │  postprocess.rs                        ││                      │
│  │   └ confidence filter, bbox 좌표 변환  │◄┘                      │
│  │     (NMS-Free: 별도 NMS 불필요)        │                       │
│  │                                        │                       │
│  │  weights: SafeTensors (~5MB)           │                       │
│  │   └ v1: fetch로 별도 로드              │                       │
│  │   └ v2: include_bytes!() 임베딩        │                       │
│  └────────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

### 추론 런타임 선택 전략

WASM으로 컴파일되어 브라우저에서 동작하기만 하면 런타임은 유연하게 선택한다.

| 런타임 | 장점 | 단점 | 우선순위 |
|--------|------|------|---------|
| **candle** (HuggingFace) | SafeTensors 네이티브, YOLOv8 WASM 예제 존재, 모델 임베딩 용이 | YOLO26 아키텍처 미구현 시 직접 작성 필요 | 1순위 시도 |
| **tract** (Sonos) | ONNX 로드 지원, WASM 호환, 순수 Rust | candle 대비 커뮤니티 작음, YOLO 예제 부족 | 2순위 대안 |
| **ort (onnxruntime-rs)** | ONNX 표준, 최적화된 추론 | WASM 타겟 지원 제한적, C++ 의존성 | 3순위 대안 |
| **rten** | 경량 ONNX runtime, WASM 지원 | 커뮤니티 소규모 | 4순위 대안 |

**결정 기준**: candle로 YOLO26 구현을 먼저 시도하고, 아키텍처 구현 난이도가 과도하면 ONNX export + tract 조합으로 전환한다.

### 모델 상세: YOLO26n Detection

| 항목 | 값 |
|------|-----|
| 파라미터 수 | 2.4M |
| FLOPs | 5.4B |
| 입력 크기 | 640x640x3 (RGB, float32) |
| mAP (COCO val) | 40.9 |
| 클래스 수 | 80 (COCO) |
| NMS | NMS-Free (end-to-end) |
| 가중치 포맷 | SafeTensors (FP16, ~5MB) 또는 ONNX (~10MB) |
| CPU 추론 (native) | 38.9ms |
| 예상 WASM 추론 | 300~800ms (single-threaded, 디바이스 의존) |

**YOLO26 선택 이유:**
- **NMS-Free** → 후처리 구현이 단순 (NMS 알고리즘 Rust 구현 불필요)
- YOLO11/v8 대비 CPU 추론 **43% 빠름**
- COCO pretrained → person(class 0) 포함 80 클래스 즉시 사용

### Rust Crate 구조

```
yolo26-rust-wasm-html-demo/
├── Cargo.toml              # candle-core, candle-nn, wasm-bindgen, serde 등
├── src/
│   ├── lib.rs              # wasm-bindgen 진입점
│   │                       #   init_model(weights: &[u8])
│   │                       #   detect(pixels: &[u8], w: u32, h: u32, threshold: f32) → JSON
│   ├── model.rs            # YOLO26n 모델 아키텍처 (candle layers)
│   │                       # 또는 ONNX 런타임 래퍼 (tract 사용 시)
│   ├── preprocess.rs       # RGBA→RGB, letterbox resize 640x640, normalize, HWC→CHW
│   └── postprocess.rs      # confidence filter, bbox 좌표 변환 (모델→원본)
├── weights/                # 모델 가중치 (gitignore 또는 Git LFS)
│   └── yolo26n.safetensors # (또는 yolo26n.onnx)
├── index.html              # 데모 UI (단일 파일)
├── pkg/                    # wasm-pack 빌드 산출물 (.gitignore)
├── PRD.md
├── .gitignore
└── README.md
```

### WASM API 인터페이스

```rust
use wasm_bindgen::prelude::*;

/// 모델 초기화 — SafeTensors/ONNX 바이트를 받아 모델을 메모리에 로드
/// 페이지 로드 시 1회 호출
#[wasm_bindgen]
pub fn init_model(weights: &[u8]) -> Result<(), JsValue>;

/// 추론 실행 — RGBA raw pixels + 원본 크기 + threshold를 받아 Detection JSON 반환
/// 이미지 디코딩은 JS Canvas API에서 수행 (WASM 바이너리 크기 절감)
#[wasm_bindgen]
pub fn detect(
    pixels: &[u8],              // RGBA raw pixels (from canvas.getImageData)
    width: u32,                 // 원본 이미지 너비
    height: u32,                // 원본 이미지 높이
    confidence_threshold: f32,  // 탐지 최소 신뢰도 (0.0~1.0)
) -> Result<String, JsValue>;  // JSON string
```

**이미지 디코딩을 JS에서 수행하는 이유:**
- Rust `image` crate를 WASM에 포함하면 바이너리 크기가 ~1MB 이상 증가
- 브라우저 Canvas API가 JPEG/PNG/WebP/HEIC 등 모든 포맷을 이미 지원
- WASM에는 순수 tensor 연산만 남겨 바이너리를 최소화

### Detection 결과 포맷

```json
{
  "detections": [
    {
      "x": 120,
      "y": 45,
      "width": 200,
      "height": 400,
      "confidence": 0.92,
      "class_id": 0,
      "class_name": "person"
    }
  ],
  "inference_time_ms": 350,
  "image_width": 1920,
  "image_height": 1080
}
```

- `x, y`: bounding box **좌상단** 좌표 (원본 이미지 기준 px)
- `width, height`: bounding box 크기 (원본 이미지 기준 px)
- `class_id`: COCO 클래스 ID (0=person, 1=bicycle, ...)
- `class_name`: COCO 클래스 이름 문자열

### 전처리 파이프라인 (preprocess.rs)

```
Input: RGBA pixels (from JS Canvas) + width + height
  │
  ├─ 1. RGBA → RGB (알파 채널 제거)
  │
  ├─ 2. Letterbox Resize → 640x640
  │     원본 종횡비 유지, 패딩 = 114/255 gray
  │     scale_factor, pad_x, pad_y 기록 (후처리에서 역변환용)
  │
  ├─ 3. Normalize: [0, 255] → [0.0, 1.0]
  │
  ├─ 4. HWC → CHW: (640, 640, 3) → (3, 640, 640)
  │
  └─ 5. Batch 추가: (3, 640, 640) → (1, 3, 640, 640)

Output: Tensor [1, 3, 640, 640] float32
```

### 후처리 파이프라인 (postprocess.rs)

YOLO26은 NMS-Free이므로 NMS 단계 불필요.

```
Input: 모델 출력 텐서 + confidence_threshold + scale_factor + pad_x/pad_y
  │
  ├─ 1. 출력 텐서 파싱: bbox (cx, cy, w, h) + class confidence 추출
  │
  ├─ 2. Confidence 필터링: max_class_confidence >= threshold 인 것만 유지
  │
  ├─ 3. 좌표 변환:
  │     cx, cy, w, h (모델 640x640 공간)
  │     → x, y, width, height (원본 이미지 공간)
  │     letterbox padding 역보정 + scale 역보정
  │
  └─ 4. JSON 직렬화: Vec<Detection> → serde_json::to_string

Output: JSON string
```

### 빌드 및 실행

```bash
# 사전 요구사항
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

# WASM 빌드
wasm-pack build --target web --release

# 로컬 실행 (WASM은 file:// 프로토콜에서 로드 불가)
npx serve .
# → http://localhost:3000/index.html
```

### 모델 가중치 준비

**방법 A: SafeTensors (candle 사용 시)**
```python
from ultralytics import YOLO
from safetensors.torch import save_file

model = YOLO("yolo26n.pt")
save_file(model.model.state_dict(), "weights/yolo26n.safetensors")
```

**방법 B: ONNX (tract 사용 시)**
```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="onnx", simplify=True, opset=17)
# → yolo26n.onnx (~10MB)
```

### HTML UI 명세

**레이아웃 (반응형)**

```
┌──────────────────────────────────────────────────────────┐
│  YOLO26 Rust WASM Demo                                    │
│  Browser-based object detection powered by Rust + WASM    │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  [📁 이미지 선택]  또는 여기에 드래그앤드롭                  │
│                                                           │
│  Confidence: 0.25 ━━━━━○━━━━━━━━━━ 1.0                   │
│                                                           │
│  ┌──────────────────────────────────────────────────┐     │
│  │                                                  │     │
│  │              Canvas                              │     │
│  │         (이미지 + bounding boxes)                 │     │
│  │                                                  │     │
│  └──────────────────────────────────────────────────┘     │
│                                                           │
│  ✓ Detected 5 objects in 342ms                            │
│                                                           │
│  ┌─ Detections ─────────────────────────────────────┐    │
│  │  #  Class     Conf    Position                   │    │
│  │  1  person    92%     (120, 45, 200, 400)        │    │
│  │  2  car       87%     (500, 200, 300, 180)       │    │
│  │  3  person    74%     (800, 50, 150, 350)        │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

**모바일 레이아웃 (< 768px)**
- 전체 요소가 세로 스택으로 배치
- Canvas가 화면 너비 100%로 확장
- 결과 목록이 Canvas 아래에 스크롤 가능하게 표시
- 이미지 선택 시 카메라 옵션도 표시 (`capture` 미지정으로 갤러리 우선)

**Bounding Box 렌더링 규칙:**
- Canvas 2D Context 사용
- Box 색상: 클래스별 고정 (HSL hue = `class_id * 137.5 % 360`, saturation 70%, lightness 50%)
- Box 스타일: `strokeRect` 2px + 반투명 fill (alpha 0.15)
- 라벨: box 상단에 배경색 rect + 흰색 텍스트 `"class_name confidence%"`
- Canvas 크기는 컨테이너에 맞춰 CSS로 조절하되, 실제 해상도는 원본 이미지 크기 유지

**사용자 흐름:**

```
1. 페이지 로드
   └→ WASM 모듈 로드 (init())
   └→ 모델 가중치 fetch → init_model(weights)
   └→ 상태: "Model loaded (X.X MB)"

2. 이미지 선택 (버튼 클릭 또는 드래그앤드롭)
   └→ JS Canvas API로 이미지 디코딩 → RGBA pixels
   └→ 상태: "Detecting..."
   └→ WASM detect(pixels, w, h, threshold) 호출
   └→ JSON 결과 파싱
   └→ Canvas에 이미지 + bounding box 렌더링
   └→ Detection 목록 업데이트
   └→ 상태: "Detected N objects in Xms"

3. Confidence 슬라이더 변경
   └→ 캐시된 전체 결과에서 threshold로 필터링 (WASM 재호출 불필요)
   └→ Canvas + 목록 즉시 업데이트
```

> **최적화 포인트**: confidence 슬라이더 변경 시 WASM을 재호출하지 않는다.
> 첫 추론 시 threshold=0.0으로 전체 결과를 받아두고, JS에서 필터링만 수행한다.
> 이렇게 하면 슬라이더 조작이 즉각적으로 반응한다.

---

## 4. Security & Privacy

| 항목 | 설명 |
|------|------|
| **데이터 전송** | 없음. 모든 이미지 처리가 브라우저 로컬에서 수행. 서버 전송 제로 |
| **모델 보호 (v1)** | SafeTensors/ONNX 파일이 별도 다운로드됨 — 네트워크 탭에서 확인 가능. 데모 단계에서는 허용 |
| **모델 보호 (v2)** | `include_bytes!()` 로 WASM에 임베딩 → 별도 파일 노출 없음. WASM 바이너리 리버스 엔지니어링은 가능하지만 난이도 높음 |
| **WASM 코드 보호** | Rust → WASM 컴파일 결과물은 바이트코드 수준이므로 JS 대비 리버스 엔지니어링 비용이 높음 |
| **CORS** | 동일 origin 정적 서빙이므로 CORS 이슈 없음 |
| **특수 헤더** | 불필요. 싱글스레드 WASM이므로 COOP/COEP 헤더 없이 동작 |

---

## 5. Risks & Roadmap

### Phase 1: MVP (이 저장소의 스코프)

> **목표**: 브라우저에서 YOLO26 WASM 추론이 동작하는 것을 검증하는 오픈소스 데모

| 항목 | 상세 |
|------|------|
| 모델 로딩 | fetch로 별도 파일 다운로드 |
| UI | 단일 `index.html`, Vanilla JS |
| 모바일 | 반응형 레이아웃, 기본 동작 보장 |
| 배포 | GitHub Pages 또는 정적 서버 |
| 라이선스 | 오픈소스 (MIT 또는 Apache 2.0) |

### Phase 2: 모델 임베딩 + 최적화

| 항목 | 상세 |
|------|------|
| `include_bytes!()` | 모델 가중치를 WASM 바이너리에 임베딩 |
| `wasm-opt` | 바이너리 크기 최적화 |
| Brotli 압축 | CDN 레벨에서 전송 크기 60~70% 절감 |
| 프로파일링 | 추론 병목 분석 및 최적화 |

### Phase 3: dontsendfile.com 통합

| 항목 | 상세 |
|------|------|
| npm 패키지화 | `wasm-pack build --target bundler` → npm 패키지 |
| Next.js 통합 | React 컴포넌트에서 WASM 모듈 호출 |
| Person Blur UI | react-konva 기반 box 편집, blur 프리뷰, 내보내기 |
| Person 전용 필터 | class_id === 0 (person)만 탐지 |

### Technical Risks

| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| **YOLO26 아키텍처를 candle에서 구현하기 어려움** | 높음 | 높음 | candle의 YOLOv8 구현 기반으로 수정 시도 → 실패 시 ONNX export + tract로 전환 |
| **SafeTensors 키 이름이 candle 모델과 불일치** | 중간 | 중간 | PyTorch state_dict 키를 덤프하여 매핑 테이블 작성 후 로드 시 변환 |
| **WASM 메모리 부족 (모바일)** | 낮음 | 중간 | JS에서 이미지를 max 4096px로 리사이즈 후 전달. WASM 메모리는 640x640 추론에 ~100MB면 충분 |
| **모바일 추론 속도가 수 초 이상** | 중간 | 낮음 | UX에 프로그레스 표시. 데모 목적이므로 속도 자체는 비차단 |
| **candle WASM 빌드 에러 (의존성 충돌)** | 중간 | 중간 | candle-wasm-examples의 Cargo.toml 버전을 그대로 따라가서 검증된 조합 사용 |
| **include_bytes 시 WASM 바이너리 > 15MB** | 낮음 | 낮음 | Phase 2에서 대응. Brotli 압축으로 전송 크기 절감. lazy loading 적용 |

---

## 6. References

### YOLO26
- [YOLO26 arXiv Paper](https://arxiv.org/html/2509.25164v1)
- [Ultralytics YOLO26 Docs](https://docs.ultralytics.com/models/yolo26/)
- [YOLO26 vs YOLO11 Comparison](https://docs.ultralytics.com/compare/yolo26-vs-yolo11/)

### Rust WASM Inference
- [candle — HuggingFace Rust ML](https://github.com/huggingface/candle)
- [candle WASM YOLO example](https://github.com/huggingface/candle/tree/main/candle-wasm-examples/yolo) — YOLOv8 브라우저 추론 레퍼런스
- [tract — Sonos ONNX runtime in Rust](https://github.com/sonos/tract)
- [rten — lightweight ONNX runtime](https://github.com/robertknight/rten)

### Build Toolchain
- [wasm-pack](https://rustwasm.github.io/wasm-pack/)
- [wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/)

### 원문
- [Discussion #43: Browser-Based Person Blur Tool](https://github.com/developer0hye/dontsendfile.com/discussions/43)

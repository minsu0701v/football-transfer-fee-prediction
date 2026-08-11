import joblib

from app.config import MODEL_FILE


# ============================================================
# 전역 모델 Bundle
# ============================================================

_model = None


# ============================================================
# Model Loader
# ============================================================

def load_model():
    """
    v1.2 앙상블 모델을 최초 한 번만 로드한다.
    """

    global _model

    if _model is None:

        if not MODEL_FILE.exists():
            raise FileNotFoundError(
                f"모델 파일을 찾을 수 없습니다: {MODEL_FILE}"
            )

        print(
            f"모델 로드 중: {MODEL_FILE.name}"
        )

        bundle = joblib.load(
            MODEL_FILE
        )

        # ====================================================
        # v1.2 Bundle 검증
        # ====================================================

        if not isinstance(bundle, dict):
            raise ValueError(
                "로드된 모델이 v1.2 앙상블 Bundle 형식이 아닙니다."
            )

        required_keys = {
            "version",
            "model_c",
            "model_d",
            "alpha_c",
            "alpha_d",
            "features_c",
            "features_d",
        }

        missing_keys = (
            required_keys
            - set(bundle.keys())
        )

        if missing_keys:
            raise ValueError(
                "v1.2 모델 Bundle에 필요한 값이 없습니다: "
                f"{sorted(missing_keys)}"
            )

        if bundle["version"] != "1.2":
            raise ValueError(
                "v1.2 모델이 아닙니다. "
                f"현재 version: {bundle['version']}"
            )

        _model = bundle

        print(
            "모델 로드 완료"
        )

        print(
            "앙상블: "
            f"C {bundle['alpha_c'] * 100:.0f}% "
            f"+ D {bundle['alpha_d'] * 100:.0f}%"
        )

        print(
            f"Model C features: "
            f"{len(bundle['features_c'])}개"
        )

        print(
            f"Model D features: "
            f"{len(bundle['features_d'])}개"
        )

    return _model


# ============================================================
# Get Model
# ============================================================

def get_model():
    """
    현재 로드된 v1.2 모델 Bundle을 반환한다.
    로드되지 않았다면 자동으로 로드한다.
    """

    return load_model()
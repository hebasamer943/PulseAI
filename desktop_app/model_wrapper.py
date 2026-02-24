from pathlib import Path
import numpy as np
import tensorflow as tf

# إذا عندك PositionalEncoding بملف ثاني، استورديه من مكانه
# مثال (عدّلي حسب مكانه الحقيقي):
from desktop_app.ecg_pipeline import PositionalEncoding

@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        self.d_model = d_model

        pos = np.arange(length)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angles = pos * angle_rates

        pe = np.zeros((length, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(angles[:, 0::2])
        pe[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.constant(pe[None, :, :])

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"length": self.length, "d_model": self.d_model})
        return cfg



class ModelWrapper:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir

        self.resnet_path = self.models_dir / "ptbdb_resnet1d_fix.keras"
        self.tcn_path = self.models_dir / "ptbdb_tcn.keras"
        self.trans_path = self.models_dir / "ptbdb_transformer_fix.keras"

        self.m_resnet = None
        self.m_tcn = None
        self.m_trans = None

        # أوزان (لأن ResNet الأفضل عندك)
        self.w_resnet = 0.50
        self.w_tcn = 0.30
        self.w_trans = 0.20

        # منطقة الشك (اختياري): إذا ResNet واثق ما منستدعي باقي النماذج
        self.unc_low = 0.45
        self.unc_high = 0.75

    def load(self):
        # نحمّل ResNet دائمًا
        if not self.resnet_path.exists():
            raise FileNotFoundError("Missing Models/ptbdb_resnet1d_fix.keras")
        self.m_resnet = tf.keras.models.load_model(self.resnet_path, compile=False)
        print("Loaded ResNet:", self.resnet_path)

        # تحميل باقي النماذج "حسب الحاجة" (رح نحمّلهم أول مرة نحتاجهم)
        if not self.tcn_path.exists():
            print("TCN missing (optional):", self.tcn_path)
        if not self.trans_path.exists():
            print("Transformer missing (optional):", self.trans_path)

    def _ensure_tcn(self):
        if self.m_tcn is None:
            if not self.tcn_path.exists():
                raise FileNotFoundError("Missing Models/ptbdb_tcn.keras")
            self.m_tcn = tf.keras.models.load_model(self.tcn_path, compile=False)
            print("Loaded TCN:", self.tcn_path)

    def _ensure_trans(self):
        if self.m_trans is None:
            if not self.trans_path.exists():
                raise FileNotFoundError("Missing Models/ptbdb_transformer_fix.keras")
            self.m_trans = tf.keras.models.load_model(
                self.trans_path,
                compile=False,
                custom_objects={"PositionalEncoding": PositionalEncoding},
            )
            print("Loaded Transformer:", self.trans_path)

    def _prep(self, beat: np.ndarray) -> np.ndarray:
        x = np.asarray(beat, dtype=np.float32)

        # تأكيد الطول 187
        if x.size > 187:
            x = x[:187]
        if x.size < 187:
            x = np.pad(x, (0, 187 - x.size))

        return x.reshape(1, 187, 1)

    def predict_proba_abn(self, beat: np.ndarray) -> float:
        if self.m_resnet is None:
            self.load()

        X = self._prep(beat)

        # Stage-1: ResNet دائمًا
        p_res = float(self.m_resnet.predict(X, verbose=0).reshape(-1)[0])

        # إذا مش ضمن منطقة الشك → نرجّع قرار ResNet مباشرة (أسرع)
        if (p_res < self.unc_low) or (p_res > self.unc_high):
            return p_res

        # ضمن منطقة الشك → نشغّل ensemble للتأكيد
        self._ensure_tcn()
        self._ensure_trans()

        p_tcn = float(self.m_tcn.predict(X, verbose=0).reshape(-1)[0])
        p_tr  = float(self.m_trans.predict(X, verbose=0).reshape(-1)[0])

        p_final = (self.w_resnet * p_res) + (self.w_tcn * p_tcn) + (self.w_trans * p_tr)
        return float(p_final)

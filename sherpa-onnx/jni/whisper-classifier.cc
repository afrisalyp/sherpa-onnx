#include <memory>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>

#include "sherpa-onnx/csrc/features.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/jni/common.h"
#include "onnxruntime_cxx_api.h"

namespace sherpa_onnx {

struct WhisperClassifier {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "whisper-classifier"};
  Ort::Session session{nullptr};
  Ort::SessionOptions session_opts;
  std::vector<std::string> labels;
};

}  // namespace sherpa_onnx

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_WhisperClassifier_newFromFile(
    JNIEnv *env, jobject /*obj*/,
    jstring model_path,
    jstring labels_path,
    jint num_threads) {

  auto *p = new sherpa_onnx::WhisperClassifier();

  p->session_opts.SetIntraOpNumThreads(num_threads);
  p->session_opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

  // Load ONNX model
  const char *model_cstr = env->GetStringUTFChars(model_path, nullptr);
  p->session = Ort::Session(p->env, model_cstr, p->session_opts);
  env->ReleaseStringUTFChars(model_path, model_cstr);

  // Load labels
  const char *labels_cstr = env->GetStringUTFChars(labels_path, nullptr);
  std::ifstream f(labels_cstr);
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty()) p->labels.push_back(line);
  }
  env->ReleaseStringUTFChars(labels_path, labels_cstr);

  return (jlong)p;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_onnx_WhisperClassifier_delete(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::WhisperClassifier *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_onnx_WhisperClassifier_compute(
    JNIEnv *env, jobject /*obj*/,
    jlong ptr,
    jfloatArray samples_array,
    jint sample_rate,
    jint top_k) {

  auto *p = reinterpret_cast<sherpa_onnx::WhisperClassifier *>(ptr);

  // ── Step 1: Ekstrak mel via FeatureExtractor sherpa-onnx ──────────────
  sherpa_onnx::FeatureExtractorConfig feat_config;
  feat_config.is_whisper = true;
  feat_config.sampling_rate = 16000;

  sherpa_onnx::FeatureExtractor extractor(feat_config);

  jsize n = env->GetArrayLength(samples_array);
  jfloat *samples = env->GetFloatArrayElements(samples_array, nullptr);
  extractor.AcceptWaveform(sample_rate, samples, n);
  extractor.InputFinished();
  env->ReleaseFloatArrayElements(samples_array, samples, JNI_ABORT);

  int32_t num_frames = extractor.NumFramesReady();
  static constexpr int32_t kNMels = 80;
  static constexpr int32_t kNFrames = 3000;

  // GetFrames returns [num_frames, 80], kita perlu [1, 80, 3000]
  int32_t frames_to_get = std::min(num_frames, kNFrames);
  std::vector<float> frames = extractor.GetFrames(0, frames_to_get);
  // frames shape: [frames_to_get, 80]

  // Transpose ke [80, 3000] + pad kalau kurang dari 3000 frames
  std::vector<float> mel(kNMels * kNFrames, 0.0f);
  for (int32_t t = 0; t < frames_to_get; ++t) {
    for (int32_t m = 0; m < kNMels; ++m) {
      mel[m * kNFrames + t] = frames[t * kNMels + m];
    }
  }

  // ── Step 2: Run ONNX model ─────────────────────────────────────────────
  auto memory_info = Ort::MemoryInfo::CreateCpu(
      OrtArenaAllocator, OrtMemTypeDefault);

  std::array<int64_t, 3> input_shape = {1, kNMels, kNFrames};
  auto input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, mel.data(), mel.size(),
      input_shape.data(), input_shape.size());

  const char *input_names[] = {"input_features"};
  const char *output_names[] = {"logits"};

  auto outputs = p->session.Run(
      Ort::RunOptions{nullptr},
      input_names, &input_tensor, 1,
      output_names, 1);

  float *logits = outputs[0].GetTensorMutableData<float>();
  int32_t num_classes = static_cast<int32_t>(p->labels.size());

  // ── Step 3: Sigmoid + Top-K ───────────────────────────────────────────
  std::vector<std::pair<float, int32_t>> scored(num_classes);
  for (int32_t i = 0; i < num_classes; ++i) {
    scored[i] = {1.0f / (1.0f + std::exp(-logits[i])), i};
  }
  int32_t k = std::min((int32_t)top_k, num_classes);
  std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                    [](auto &a, auto &b) { return a.first > b.first; });

  // ── Step 4: Return ke Kotlin sebagai array AudioEvent ─────────────────
  jclass cls = env->FindClass("com/k2fsa/sherpa/onnx/AudioEvent");
  jmethodID ctor = env->GetMethodID(cls, "<init>", "(Ljava/lang/String;IF)V");
  jobjectArray result = env->NewObjectArray(k, cls, nullptr);

  for (int32_t i = 0; i < k; ++i) {
    jstring name = env->NewStringUTF(p->labels[scored[i].second].c_str());
    jobject event = env->NewObject(cls, ctor, name, scored[i].second, scored[i].first);
    env->SetObjectArrayElement(result, i, event);
    env->DeleteLocalRef(name);
    env->DeleteLocalRef(event);
  }
  env->DeleteLocalRef(cls);

  return result;
}

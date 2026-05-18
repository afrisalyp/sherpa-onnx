// sherpa-onnx/jni/whisper-classifier.cc
#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/features.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/jni/common.h"
#include "onnxruntime_cxx_api.h"

namespace sherpa_onnx {

struct WhisperClassifier {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "whisper-classifier"};
  Ort::Session session{nullptr};
  Ort::SessionOptions session_opts;
  std::vector<std::string> labels;

  void InitSession(const std::vector<char> &model_bytes, int32_t num_threads) {
    session_opts.SetIntraOpNumThreads(num_threads);
    session_opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session = Ort::Session(env,
                           model_bytes.data(),
                           model_bytes.size(),
                           session_opts);
  }

  void InitLabels(const std::string &content) {
    std::istringstream is(content);
    std::string line;
    while (std::getline(is, line)) {
      if (!line.empty()) labels.push_back(line);
    }
  }
};

}  // namespace sherpa_onnx

// ── newFromFile ────────────────────────────────────────────────────────────
SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_WhisperClassifier_newFromFile(
    JNIEnv *env, jobject /*obj*/,
    jstring model_path,
    jstring labels_path,
    jint num_threads) {

  auto *p = new sherpa_onnx::WhisperClassifier();

  const char *model_cstr = env->GetStringUTFChars(model_path, nullptr);
  auto model_bytes = sherpa_onnx::ReadFile(model_cstr);
  env->ReleaseStringUTFChars(model_path, model_cstr);
  p->InitSession(model_bytes, num_threads);

  const char *labels_cstr = env->GetStringUTFChars(labels_path, nullptr);
  auto labels_bytes = sherpa_onnx::ReadFile(labels_cstr);
  env->ReleaseStringUTFChars(labels_path, labels_cstr);
  p->InitLabels(std::string(labels_bytes.data(), labels_bytes.size()));

  return (jlong)p;
}

// ── newFromAsset ───────────────────────────────────────────────────────────
SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_WhisperClassifier_newFromAsset(
    JNIEnv *env, jobject /*obj*/,
    jobject asset_manager,
    jstring model_path,
    jstring labels_path,
    jint num_threads) {

#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    return 0;
  }

  auto *p = new sherpa_onnx::WhisperClassifier();

  const char *model_cstr = env->GetStringUTFChars(model_path, nullptr);
  auto model_bytes = sherpa_onnx::ReadFile(mgr, model_cstr);
  env->ReleaseStringUTFChars(model_path, model_cstr);
  p->InitSession(model_bytes, num_threads);

  const char *labels_cstr = env->GetStringUTFChars(labels_path, nullptr);
  auto labels_bytes = sherpa_onnx::ReadFile(mgr, labels_cstr);
  env->ReleaseStringUTFChars(labels_path, labels_cstr);
  p->InitLabels(std::string(labels_bytes.data(), labels_bytes.size()));

  return (jlong)p;
#else
  return 0;
#endif
}

// ── delete ─────────────────────────────────────────────────────────────────
SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_onnx_WhisperClassifier_delete(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::WhisperClassifier *>(ptr);
}

// ── compute ────────────────────────────────────────────────────────────────
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

  static constexpr int32_t kNMels = 80;
  static constexpr int32_t kNFrames = 3000;

  int32_t num_frames = extractor.NumFramesReady();
  int32_t frames_to_get = std::min(num_frames, kNFrames);

  // GetFrames returns [frames_to_get, 80], transpose ke [80, 3000]
  std::vector<float> frames = extractor.GetFrames(0, frames_to_get);
  // std::vector<float> mel(kNMels * kNFrames, 0.0f);
  // for (int32_t t = 0; t < frames_to_get; ++t) {
  //   for (int32_t m = 0; m < kNMels; ++m) {
  //     mel[m * kNFrames + t] = frames[t * kNMels + m];
  //   }
  // }

  // Transpose ke [80, 3000]
  std::vector<float> mel(kNMels * kNFrames, 0.0f);
  for (int32_t t = 0; t < frames_to_get; ++t) {
    for (int32_t m = 0; m < kNMels; ++m) {
      mel[m * kNFrames + t] = frames[t * kNMels + m];
    }
  }

  // ── Whisper normalisasi (identik dengan WhisperProcessor HuggingFace) ──
  // log10
  for (auto &v : mel) {
    v = std::log10(std::max(v, 1e-10f));
  }

  // max(log_mel, log_mel.max() - 8.0)
  float mel_max = *std::max_element(mel.begin(), mel.end());
  for (auto &v : mel) {
    v = std::max(v, mel_max - 8.0f);
  }

  // (log_mel + 4.0) / 4.0
  for (auto &v : mel) {
    v = (v + 4.0f) / 4.0f;
  }

  // Debug: log beberapa nilai mel
  SHERPA_ONNX_LOGE("StutterMate debug: mel min/max/mean check:");
  float mel_min = *std::min_element(mel.begin(), mel.end());
  // float mel_max = *std::max_element(mel.begin(), mel.end());
  float mel_sum = 0;
  for (auto v : mel) mel_sum += v;
  SHERPA_ONNX_LOGE("StutterMate debug: min=%.4f max=%.4f mean=%.4f", 
                    mel_min, mel_max, mel_sum / mel.size());
  SHERPA_ONNX_LOGE("StutterMate debug: mel[0*3000+0]=%.4f mel[0*3000+1]=%.4f mel[0*3000+2]=%.4f",
                    mel[0], mel[1], mel[2]);

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

  // ── Step 3: Softmax + Top-K ───────────────────────────────────────────
  std::vector<float> probs(num_classes);
  float max_logit = *std::max_element(logits, logits + num_classes);
  float sum_exp = 0.0f;
  for (int32_t i = 0; i < num_classes; ++i) {
    probs[i] = std::exp(logits[i] - max_logit);  // subtract max untuk numerical stability
    sum_exp += probs[i];
  }
  for (int32_t i = 0; i < num_classes; ++i) {
    probs[i] /= sum_exp;
  }
  std::vector<std::pair<float, int32_t>> scored(num_classes);
  for (int32_t i = 0; i < num_classes; ++i) {
    scored[i] = {probs[i], i};
  }

  int32_t k = std::min(top_k, num_classes);
  std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                    [](auto &a, auto &b) { return a.first > b.first; });

  // ── Step 4: Return ke Kotlin sebagai array AudioEvent ─────────────────
  jclass cls = env->FindClass("com/k2fsa/sherpa/onnx/AudioEvent");
  jmethodID ctor = env->GetMethodID(cls, "<init>", "(Ljava/lang/String;IF)V");
  jobjectArray result = env->NewObjectArray(k, cls, nullptr);

  for (int32_t i = 0; i < k; ++i) {
    jstring name = env->NewStringUTF(p->labels[scored[i].second].c_str());
    jobject event = env->NewObject(cls, ctor, name, scored[i].second,
                                   scored[i].first);
    env->SetObjectArrayElement(result, i, event);
    env->DeleteLocalRef(name);
    env->DeleteLocalRef(event);
  }
  env->DeleteLocalRef(cls);

  return result;
}
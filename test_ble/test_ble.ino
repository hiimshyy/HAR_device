#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model_int8.h"

namespace{
  bool isCapturing = false;
  int8_t attempt = 0;
  int8_t samplesNum = 120;
  int8_t samplesRead = samplesNum;
  float accThreshold = 2.0;
  float aX, aY, aZ, gX, gY, gZ;
  String myValue = "unknown";

  // Bluetooth® Low Energy Battery Service
  BLEService fitnessService("b7fb8e6c-0000-4ee6-9dc8-9c45b99a0356");
  BLEStringCharacteristic modelCharacteristic("b7fb8e6c-8000-4ee6-9dc8-9c45b99a0356", BLENotify, 255);

  BLEDevice central;

  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  const tflite::Model* tflModel = nullptr;
  tflite::MicroInterpreter* tflInterpreter = nullptr;
  TfLiteTensor* tflInputTensor = nullptr;
  TfLiteTensor* tflOutputTensor = nullptr;

  // global variables used for TensorFlow Lite (Micro)
  tflite::ErrorReporter* error_reporter = nullptr;

  // // pull in all the TFLM ops, you can remove this line and
  // // only pull in the TFLM ops you need, if would like to reduce
  // // the compiled size of the sketch.
  //tflite::AllOpsResolver tflOpsResolver;

  // Create a static memory buffer for TFLM, the size may need to
  // be adjusted based on the model you are using
  constexpr int tensorArenaSize = 25 * 1024;
  byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

  constexpr int NUM_LABELS = 4;
  // array to map gesture index to a name
  const char* LABELS[NUM_LABELS] = {
    "SitDown",
    "StandUp",
    "Jump",
    "RotateLeft"
  };

} // namespace

void setup() {
  Serial.begin(9600);    // initialize serial communication

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  // begin initialization
  if (!BLE.begin()) {
    Serial.println("starting BLE failed!");
    while (1);
  }

  BLE.setLocalName("Nano33BLE");
  BLE.setAdvertisedService(fitnessService); // add the service UUID
  fitnessService.addCharacteristic(modelCharacteristic); // add the battery level characteristic
  BLE.addService(fitnessService); // Add the battery service
  BLE.advertise();

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  static tflite::MicroErrorReporter micro_error_reporter;  // NOLINT
  error_reporter = &micro_error_reporter;
  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model_int8);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         tflModel->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  static tflite::MicroMutableOpResolver<8> tflOpsResolver;  // NOLINT
  tflOpsResolver.AddQuantize();
  tflOpsResolver.AddConv2D();
  tflOpsResolver.AddMaxPool2D();
  tflOpsResolver.AddFullyConnected();
  tflOpsResolver.AddReshape();
  tflOpsResolver.AddSoftmax();
  tflOpsResolver.AddDequantize();
  tflOpsResolver.AddExpandDims();
  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, error_reporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  central = BLE.central();

  if (central){
    digitalWrite(LED_BUILTIN, HIGH);
    while (central.connected()){
      if (modelCharacteristic.subscribed()){
        getData();
        //timer();
      }
    }
    digitalWrite(LED_BUILTIN, LOW);
  }
}
void getData(){
  while (samplesRead == samplesNum) {
    if (IMU.accelerationAvailable()) {
     
      IMU.readAcceleration(aX, aY, aZ);

      float average = fabs(aX ) + fabs(aY) + fabs(aZ);

      if (average >= accThreshold) {
        samplesRead = 0;
        break;
      }
    }
  }
  while (samplesRead < samplesNum) {
    // make sure IMU data is available then read in data
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      // Quantize accelerometer data
      tflInputTensor->data.int8[samplesRead * 6 + 0] = (aX / 8.0) * 255;
      tflInputTensor->data.int8[samplesRead * 6 + 1] = (aY / 8.0) * 255;
      tflInputTensor->data.int8[samplesRead * 6 + 2] = (aZ / 8.0) * 255;

      // Quantize gyroscope data
      tflInputTensor->data.int8[samplesRead * 6 + 3] = (gX / 8.0) * 255;
      tflInputTensor->data.int8[samplesRead * 6 + 4] = (gY / 8.0) * 255;
      tflInputTensor->data.int8[samplesRead * 6 + 5] = (gZ / 8.0) * 255;

      samplesRead++;
    }
  }

        TfLiteStatus invoke_status = tflInterpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
          TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
          return;
         };
  if (samplesRead == samplesNum) {
        int8_t max_score;
        int max_index;
        ++attempt;
        Serial.print("Attempt: ");
        Serial.println(attempt);
        for (int i = 0; i < NUM_LABELS; ++i) {
          const int8_t score = tflOutputTensor->data.int8[i];
          if ((i == 0) || (score > max_score)) {
            max_score = score;
            max_index = i;
          } 
        }
        TF_LITE_REPORT_ERROR(error_reporter, "Found %s (%d)", LABELS[max_index], max_score);
        myValue = String(max_index);
        modelCharacteristic.setValue(myValue);
      }
}
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

#include "nvs_flash.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"

#include "esp_http_server.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"


// ============================================================
// Generated model_data.cpp
// ============================================================

extern unsigned char agnews_transformer_fp32_tflite[];
extern unsigned int agnews_transformer_fp32_tflite_len;


// ============================================================
// Configuration
// ============================================================

#define WIFI_SSID       "YOUR_WIFI"
#define WIFI_PASSWORD   "YOUR_PASSWORD"

#define MAX_LEN         64

static const char *TAG = "AGNEWS";


// ============================================================
// Tensor arena
// ============================================================

constexpr size_t TENSOR_ARENA_SIZE = 5 * 1024 * 1024;

static uint8_t *tensor_arena;


// ============================================================
// TFLite objects
// ============================================================

static const tflite::Model *model;

static tflite::MicroInterpreter *interpreter;

static TfLiteTensor *input_ids;

static TfLiteTensor *attention_mask;

static TfLiteTensor *output;


// ============================================================
// HTML
// ============================================================

static const char HTML[] = R"html(
<!DOCTYPE html>
<html>

<head>

<meta name="viewport"
      content="width=device-width,initial-scale=1">

<title>ESP32 AG News</title>

<style>

body {
    font-family: Arial;
    max-width: 700px;
    margin: 40px auto;
    padding: 20px;
}

textarea {
    width: 100%;
    height: 180px;
    padding: 10px;
    font-size: 16px;
    box-sizing: border-box;
}

button {
    margin-top: 10px;
    padding: 12px 20px;
    font-size: 16px;
}

#result {
    margin-top: 20px;
    padding: 15px;
    background: #eee;
}

</style>

</head>

<body>

<h1>ESP32-S3 AG News</h1>

<textarea id="text"
placeholder="Enter news text..."></textarea>

<br>

<button onclick="predict()">
Predict
</button>

<div id="result">
Waiting...
</div>

<script>

async function predict() {

    const text =
        document.getElementById("text").value;

    document.getElementById("result").innerText =
        "Running inference...";

    const response = await fetch(
        "/predict",
        {
            method: "POST",

            headers: {
                "Content-Type":
                    "application/json"
            },

            body: JSON.stringify({
                text: text
            })
        }
    );

    const result =
        await response.json();

    document.getElementById("result").innerHTML =

        "<h2>" + result.label + "</h2>" +

        "<p>World: " +
        result.scores[0].toFixed(4) +
        "</p>" +

        "<p>Sports: " +
        result.scores[1].toFixed(4) +
        "</p>" +

        "<p>Business: " +
        result.scores[2].toFixed(4) +
        "</p>" +

        "<p>Sci/Tech: " +
        result.scores[3].toFixed(4) +
        "</p>" +

        "<p>Inference: " +
        result.time_ms +
        " ms</p>";
}

</script>

</body>

</html>
)html";


// ============================================================
// WiFi
// ============================================================

static void wifi_event_handler(
    void *arg,
    esp_event_base_t event_base,
    int32_t event_id,
    void *event_data
) {

    if (
        event_base == WIFI_EVENT &&
        event_id == WIFI_EVENT_STA_START
    ) {

        esp_wifi_connect();

    }

    else if (
        event_base == WIFI_EVENT &&
        event_id == WIFI_EVENT_STA_DISCONNECTED
    ) {

        esp_wifi_connect();

    }

    else if (
        event_base == IP_EVENT &&
        event_id == IP_EVENT_STA_GOT_IP
    ) {

        auto *event =
            (ip_event_got_ip_t *)event_data;

        ESP_LOGI(
            TAG,
            "IP: " IPSTR,
            IP2STR(&event->ip_info.ip)
        );
    }
}


static void wifi_init()
{

    ESP_ERROR_CHECK(
        esp_netif_init()
    );

    ESP_ERROR_CHECK(
        esp_event_loop_create_default()
    );

    esp_netif_create_default_wifi_sta();


    wifi_init_config_t cfg =
        WIFI_INIT_CONFIG_DEFAULT();

    ESP_ERROR_CHECK(
        esp_wifi_init(&cfg)
    );


    ESP_ERROR_CHECK(
        esp_event_handler_register(
            WIFI_EVENT,
            ESP_EVENT_ANY_ID,
            wifi_event_handler,
            nullptr
        )
    );

    ESP_ERROR_CHECK(
        esp_event_handler_register(
            IP_EVENT,
            IP_EVENT_STA_GOT_IP,
            wifi_event_handler,
            nullptr
        )
    );


    wifi_config_t config = {};

    strcpy(
        (char *)config.sta.ssid,
        WIFI_SSID
    );

    strcpy(
        (char *)config.sta.password,
        WIFI_PASSWORD
    );


    ESP_ERROR_CHECK(
        esp_wifi_set_mode(
            WIFI_MODE_STA
        )
    );

    ESP_ERROR_CHECK(
        esp_wifi_set_config(
            WIFI_IF_STA,
            &config
        )
    );

    ESP_ERROR_CHECK(
        esp_wifi_start()
    );
}


// ============================================================
// Model initialization
// ============================================================

static void model_init()
{

    ESP_LOGI(
        TAG,
        "Model size: %u bytes",
        agnews_transformer_fp32_tflite_len
    );


    // Load model

    model =
        tflite::GetModel(
            agnews_transformer_fp32_tflite
        );


    if (
        model->version() !=
        TFLITE_SCHEMA_VERSION
    ) {

        ESP_LOGE(
            TAG,
            "Wrong TFLite schema version"
        );

        abort();
    }


    // --------------------------------------------------------
    // Resolver
    // --------------------------------------------------------

    static tflite::MicroMutableOpResolver<32>
        resolver;


    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddSub();

    resolver.AddDiv();

    resolver.AddFullyConnected();

    resolver.AddGather();

    resolver.AddMatMul();

    resolver.AddMean();

    resolver.AddRsqrt();

    resolver.AddSoftmax();

    resolver.AddReshape();

    resolver.AddTranspose();

    resolver.AddExpandDims();

    resolver.AddSqueeze();

    resolver.AddCast();

    resolver.AddShape();

    resolver.AddStridedSlice();

    resolver.AddPack();


    // --------------------------------------------------------
    // Interpreter
    // --------------------------------------------------------

    static tflite::MicroInterpreter
        static_interpreter(

            model,

            resolver,

            tensor_arena,

            TENSOR_ARENA_SIZE

        );


    interpreter =
        &static_interpreter;


    // Allocate tensors

    if (
        interpreter->AllocateTensors()
        != kTfLiteOk
    ) {

        ESP_LOGE(
            TAG,
            "AllocateTensors failed"
        );

        abort();
    }


    // Inputs

    input_ids =
        interpreter->input(0);

    attention_mask =
        interpreter->input(1);


    // Output

    output =
        interpreter->output(0);


    ESP_LOGI(
        TAG,
        "Model initialized"
    );

    ESP_LOGI(
        TAG,
        "Input 0 type: %d",
        input_ids->type
    );

    ESP_LOGI(
        TAG,
        "Input 1 type: %d",
        attention_mask->type
    );

    ESP_LOGI(
        TAG,
        "Output type: %d",
        output->type
    );
}


// ============================================================
// Temporary input
// ============================================================

static void prepare_input()
{

    // Clear

    for (int i = 0; i < MAX_LEN; i++) {

        input_ids->data.i32[i] = 0;

        attention_mask->data.i32[i] = 0;
    }


    // [CLS]
    input_ids->data.i32[0] = 101;

    attention_mask->data.i32[0] = 1;


    // [SEP]

    input_ids->data.i32[1] = 102;

    attention_mask->data.i32[1] = 1;
}


// ============================================================
// Inference
// ============================================================

static bool run_inference()
{

    return
        interpreter->Invoke()
        == kTfLiteOk;
}


// ============================================================
// GET /
// ============================================================

static esp_err_t root_handler(
    httpd_req_t *req
)
{

    httpd_resp_set_type(
        req,
        "text/html"
    );

    return httpd_resp_send(
        req,
        HTML,
        HTTPD_RESP_USE_STRLEN
    );
}


// ============================================================
// POST /predict
// ============================================================

static esp_err_t predict_handler(
    httpd_req_t *req
)
{

    char request[1024];

    int len =
        httpd_req_recv(
            req,
            request,
            sizeof(request) - 1
        );


    if (len <= 0) {

        return ESP_FAIL;
    }


    request[len] = '\0';


    ESP_LOGI(
        TAG,
        "Request: %s",
        request
    );


    // Temporary fixed input

    prepare_input();


    int64_t start =
        esp_timer_get_time();


    bool success =
        run_inference();


    int64_t end =
        esp_timer_get_time();


    if (!success) {

        httpd_resp_send_err(
            req,
            HTTPD_500_INTERNAL_SERVER_ERROR,
            "Inference failed"
        );

        return ESP_FAIL;
    }


    float *scores =
        output->data.f;


    int best = 0;


    for (int i = 1; i < 4; i++) {

        if (
            scores[i] >
            scores[best]
        ) {

            best = i;
        }
    }


    const char *labels[] = {

        "World",
        "Sports",
        "Business",
        "Sci/Tech"

    };


    char response[512];


    snprintf(

        response,

        sizeof(response),

        "{"
        "\"label\":\"%s\","
        "\"scores\":["
        "%.6f,"
        "%.6f,"
        "%.6f,"
        "%.6f"
        "],"
        "\"time_ms\":%.2f"
        "}",

        labels[best],

        scores[0],
        scores[1],
        scores[2],
        scores[3],

        (double)(end - start) / 1000.0
    );


    httpd_resp_set_type(
        req,
        "application/json"
    );


    return httpd_resp_send(
        req,
        response,
        HTTPD_RESP_USE_STRLEN
    );
}


// ============================================================
// Web server
// ============================================================

static void start_webserver()
{

    httpd_config_t config =
        HTTPD_DEFAULT_CONFIG();


    httpd_handle_t server = nullptr;


    ESP_ERROR_CHECK(
        httpd_start(
            &server,
            &config
        )
    );


    httpd_uri_t root = {

        .uri = "/",
        .method = HTTP_GET,
        .handler = root_handler,
        .user_ctx = nullptr

    };


    httpd_register_uri_handler(
        server,
        &root
    );


    httpd_uri_t predict = {

        .uri = "/predict",
        .method = HTTP_POST,
        .handler = predict_handler,
        .user_ctx = nullptr

    };


    httpd_register_uri_handler(
        server,
        &predict
    );
}


// ============================================================
// Main
// ============================================================

extern "C"
void app_main()
{

    ESP_ERROR_CHECK(
        nvs_flash_init()
    );


    // WiFi

    wifi_init();


    // Tensor arena

    tensor_arena =
        (uint8_t *)heap_caps_malloc(

            TENSOR_ARENA_SIZE,

            MALLOC_CAP_SPIRAM |
            MALLOC_CAP_8BIT
        );


    if (!tensor_arena) {

        ESP_LOGE(
            TAG,
            "Could not allocate tensor arena"
        );

        abort();
    }


    // Model

    model_init();


    // HTTP

    start_webserver();


    ESP_LOGI(
        TAG,
        "Server ready"
    );
}

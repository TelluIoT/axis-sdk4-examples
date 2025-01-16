/**
 * Copyright (C) 2021 Axis Communications AB, Lund, Sweden
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * - object_detection -
 *
 * This application loads a larod model which takes an image as input and
 * outputs values corresponding to the class, score and location of detected
 * objects in the image.
 *
 * The application expects eight arguments on the command line in the
 * following order: MODEL WIDTH HEIGHT QUALITY RAW_WIDTH RAW_HEIGHT
 * THRESHOLD LABELSFILE.
 *
 * First argument, MODEL, is a string describing path to the model.
 *
 * Second argument, WIDTH, is an integer for the input width.
 *
 * Third argument, HEIGHT, is an integer for the input height.
 *
 * Fourth argument, QUALITY, is an integer for the desired jpeg quality.
 *
 * Fifth argument, RAW_WIDTH is an integer for camera width resolution.
 *
 * Sixth argument, RAW_HEIGHT is an integer for camera height resolution.
 *
 * Seventh argument, THRESHOLD is an integer ranging from 0 to 100 to select good detections.
 *
 * Eighth argument, LABELSFILE, is a string describing path to the label txt.
 *
 */

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <syslog.h>
#include <unistd.h>

#include "argparse.h"
#include "imgprovider.h"
#include "imgutils.h"
#include "larod.h"
#include "vdo-frame.h"
#include "vdo-types.h"

#define SLEEP_PERIOD_MS 2000

// FOR AXOVERLAY
#ifdef ENABLE_OVERLAY
#include <axoverlay.h>
#include <cairo/cairo.h>
#include <math.h>
#endif

#ifdef ENABLE_CV25_OVERLAY
#include <bbox.h>
#include <math.h>
#endif

// GLOBALS OBJDETECTION
// TODO: COPIED OUT OF MAIN
// Hardcode to use three image "color" channels (eg. RGB).
const unsigned int CHANNELS = 3;
// Hardcode to set output bytes of four tensors from MobileNet V2 model.
const unsigned int FLOATSIZE   = 4;
const unsigned int TENSOR1SIZE = 80 * FLOATSIZE;
const unsigned int TENSOR2SIZE = 20 * FLOATSIZE;
const unsigned int TENSOR3SIZE = 20 * FLOATSIZE;
const unsigned int TENSOR4SIZE = 1 * FLOATSIZE;

// Name patterns for the temp file we will create.

// Pre-processing of the High resolution frame input and output
char PP_HD_INPUT_FILE_PATTERN[]  = "/tmp/larod.pp.hd.test-XXXXXX";
char PP_HD_OUTPUT_FILE_PATTERN[] = "/tmp/larod.pp.hd.out.test-XXXXXX";

// Pre-processing of the Low resolution frame input and output
// The output of the pre-processing correspond with the input of the object detector
char PP_SD_INPUT_FILE_PATTERN[]           = "/tmp/larod.pp.test-XXXXXX";
char OBJECT_DETECTOR_INPUT_FILE_PATTERN[] = "/tmp/larod.in.test-XXXXXX";

char OBJECT_DETECTOR_OUT1_FILE_PATTERN[] = "/tmp/larod.out1.test-XXXXXX";
char OBJECT_DETECTOR_OUT2_FILE_PATTERN[] = "/tmp/larod.out2.test-XXXXXX";
char OBJECT_DETECTOR_OUT3_FILE_PATTERN[] = "/tmp/larod.out3.test-XXXXXX";
char OBJECT_DETECTOR_OUT4_FILE_PATTERN[] = "/tmp/larod.out4.test-XXXXXX";

bool ret                        = false;
ImgProvider_t* sdImageProvider  = NULL;
ImgProvider_t* hdImageProvider  = NULL;
larodError* error               = NULL;
larodConnection* conn           = NULL;
larodMap* ppMap                 = NULL;
larodMap* cropMap               = NULL;
larodMap* ppMapHD               = NULL;
larodModel* ppModel             = NULL;
larodModel* ppModelHD           = NULL;
larodModel* model               = NULL;
larodTensor** ppInputTensors    = NULL;
size_t ppNumInputs              = 0;
larodTensor** ppOutputTensors   = NULL;
size_t ppNumOutputs             = 0;
larodTensor** ppInputTensorsHD  = NULL;
size_t ppNumInputsHD            = 0;
larodTensor** ppOutputTensorsHD = NULL;
size_t ppNumOutputsHD           = 0;
larodTensor** inputTensors      = NULL;
size_t numInputs                = 0;
larodTensor** outputTensors     = NULL;
size_t numOutputs               = 0;
larodJobRequest* ppReq          = NULL;
larodJobRequest* ppReqHD        = NULL;
larodJobRequest* infReq         = NULL;
void* cropAddr                  = NULL;
void* ppInputAddr               = MAP_FAILED;
void* ppInputAddrHD             = MAP_FAILED;
void* ppOutputAddrHD            = MAP_FAILED;
void* larodInputAddr            = MAP_FAILED;  // this address is both used for the output of the
                                               // preprocessing and input for the inference
void* larodOutput1Addr = MAP_FAILED;
void* larodOutput2Addr = MAP_FAILED;
void* larodOutput3Addr = MAP_FAILED;
void* larodOutput4Addr = MAP_FAILED;
int larodModelFd       = -1;
int ppInputFd          = -1;
int ppInputFdHD        = -1;
int ppOutputFdHD       = -1;
int larodInputFd       = -1;  // This file descriptor is used for both as output for the pre
                              // processing and input for the inference
int larodOutput1Fd = -1;
int larodOutput2Fd = -1;
int larodOutput3Fd = -1;
int larodOutput4Fd = -1;
char** labels      = NULL;   // This is the array of label strings. The label
                             // entries points into the large labelFileData buffer.
size_t numLabels    = 0;     // Number of entries in the labels array.
char* labelFileData = NULL;  // Buffer holding the complete collection of label strings.

// DECLARATIONS COPIED OUT OF MAIN 2
unsigned int widthFrameHD;
unsigned int heightFrameHD;

size_t yuyvBufferSize;

// ARGS
char* chipString;
char* modelFile;
char* labelsFile;
int inputWidth;
int inputHeight;
int desiredHDImgWidth;
int desiredHDImgHeight;
int threshold;
int quality;

#ifdef ENABLE_CV25_OVERLAY
bbox_t* overlay = NULL;

bbox_color_t BOUNDING_BOX_COLOR_RED;
bbox_color_t BOUNDING_BOX_COLOR_GREEN;
bbox_color_t BOUNDING_BOX_COLOR_BLUE;
bbox_color_t BOUNDING_BOX_COLOR_BLACK;
#endif

// AXOVERLAY
#if defined(ENABLE_OVERLAY) || defined(ENABLE_CV25_OVERLAY)
#define OVERLAY_SCORE_THRESHOLD 0.2

// ------------------------------------------------------------------------------------------------------------------------
// GLOBALS AXOVERLAY
#define PALETTE_VALUE_RANGE 255.0

typedef struct {
    gint top;
    gint bottom;
    gint left;
    gint right;
    char class[50];
    float score;
    gint bounding_box_id;
    gint text_id;
#ifdef ENABLE_OVERLAY
    struct axoverlay_overlay_data bounding_box;
    struct axoverlay_overlay_data text;
#endif
} ObjectOverlay;

#define OBJECT_OVERLAYS_MAX_LENGTH 5
size_t object_overlays_length = 0;
ObjectOverlay object_overlays[5];

// static gint animation_timer = -1;
// static gint overlay_id      = -1;
// static gint overlay_id_text = -1;
// static gint counter = 10;
// static gint top_color = 1;
// static gint bottom_color    = 3;

// HACKY ADDED GLOBALS
// static gboolean has_object = FALSE;
// static gint object_top    = 0;
// static gint object_bottom = 0;
// static gint object_left   = 0;
// static gint object_right  = 0;

// static gint object_top_2    = 0;
// static gint object_bottom_2 = 0;
// static gint object_left_2   = 0;
// static gint object_right_2  = 0;

// TODO: these end up being set in some callback function atm .super hacky
static gint stream_width  = 1280;
static gint stream_height = 720;

static void get_coordinates(int* out_top,
                            int* out_left,
                            int* out_bottom,
                            int* out_right,
                            unsigned int frame_width,
                            unsigned int frame_height,
                            float top,
                            float left,
                            float bottom,
                            float right) {
    unsigned int croppedWidthHD = heightFrameHD;
    unsigned int crop_x         = left * croppedWidthHD + (widthFrameHD - heightFrameHD) / 2;
    unsigned int crop_y         = top * heightFrameHD;
    unsigned int crop_w         = (right - left) * croppedWidthHD;
    unsigned int crop_h         = (bottom - top) * heightFrameHD;

    *out_top    = crop_y;
    *out_left   = crop_x;
    *out_bottom = crop_y + crop_h;
    *out_right  = crop_x + crop_w;

    // *out_top    = crop_y;           // round(locations[4 * 0] * desiredHDImgHeight);
    // *out_left   = crop_x + crop_w;  // crop_round(locations[4 * 0 + 1] * desiredHDImgWidth);
    // *out_bottom = crop_y + crop_h;  // round(locations[4 * 0 + 2] * desiredHDImgHeight);
    // *out_right  = crop_x;           // round(locations[4 * 0 + 3] * desiredHDImgWidth);

    // TESTING TO SCALE BY HEIGHT/WIDTH
    // *out_top    = round(top * frame_height);
    // *out_left   = round(left * frame_width);
    // *out_bottom = round(bottom * frame_height);
    // *out_right  = round(right * frame_width);

    syslog(LOG_INFO,
           "Width: %d Height %d Top: %f->%d Left: %f->%d Bottom: %f->%d Right: %f->%d",
           frame_width,
           frame_height,
           top,
           *out_top,
           left,
           *out_left,
           bottom,
           *out_bottom,
           right,
           *out_right);
}
#endif

#ifdef ENABLE_OVERLAY
/***** Drawing functions *****************************************************/

/**
 * brief Converts palette color index to cairo color value.
 *
 * This function converts the palette index, which has been initialized by
 * function axoverlay_set_palette_color to a value that can be used by
 * function cairo_set_source_rgba.
 *
 * param color_index Index in the palette setup.
 *
 * return color value.
 */
static gdouble index2cairo(const gint color_index) {
    return ((color_index << 4) + color_index) / PALETTE_VALUE_RANGE;
}

/**
 * brief Draw a rectangle using palette.
 *
 * This function draws a rectangle with lines from coordinates
 * left, top, right and bottom with a palette color index and
 * line width.
 *
 * param context Cairo rendering context.
 * param left Left coordinate (x1).
 * param top Top coordinate (y1).
 * param right Right coordinate (x2).
 * param bottom Bottom coordinate (y2).
 * param color_index Palette color index.
 * param line_width Rectange line width.
 */
static void draw_rectangle(cairo_t* context,
                           gint left,
                           gint top,
                           gint right,
                           gint bottom,
                           gint color_index,
                           gint line_width) {
    gdouble val = 0;

    val = index2cairo(color_index);
    cairo_set_source_rgba(context, val, val, val, val);
    cairo_set_operator(context, CAIRO_OPERATOR_SOURCE);
    cairo_set_line_width(context, line_width);
    cairo_rectangle(context, left, top, right - left, bottom - top);
    cairo_stroke(context);
}

/**
 * brief Draw a text using cairo.
 *
 * This function draws a text with a specified middle position,
 * which will be adjusted depending on the text length.
 *
 * param context Cairo rendering context.
 * param pos_x Center position coordinate (x).
 * param pos_y Center position coordinate (y).
 */
static void draw_text(cairo_t* context, char* string, const gint pos_x, const gint pos_y) {
    cairo_text_extents_t te;
    cairo_text_extents_t te_length;

    //  Show text in black
    cairo_set_source_rgb(context, 0, 0, 0);
    cairo_select_font_face(context, "serif", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(context, 15.0);

    // Position the text at a fix centered position
    cairo_text_extents(context, string, &te_length);
    cairo_move_to(context, pos_x - te_length.width / 2, pos_y);

    // Add the counter number to the shown text
    cairo_text_extents(context, string, &te);
    cairo_show_text(context, string);
}

/**
 * brief Setup an overlay_data struct.
 *
 * This function initialize and setup an overlay_data
 * struct with default values.
 *
 * param data The overlay data struct to initialize.
 */
static void setup_axoverlay_data(struct axoverlay_overlay_data* data) {
    axoverlay_init_overlay_data(data);
    data->postype         = AXOVERLAY_CUSTOM_NORMALIZED;
    data->anchor_point    = AXOVERLAY_ANCHOR_CENTER;
    data->x               = 0.0;
    data->y               = 0.0;
    data->scale_to_stream = FALSE;
}

/**
 * brief Setup palette color table.
 *
 * This function initialize and setup an palette index
 * representing ARGB values.
 *
 * param color_index Palette color index.
 * param r R (red) value.
 * param g G (green) value.
 * param b B (blue) value.
 * param a A (alpha) value.
 *
 * return result as boolean
 */
static gboolean
setup_palette_color(const int index, const gint r, const gint g, const gint b, const gint a) {
    GError* error = NULL;
    struct axoverlay_palette_color color;

    color.red      = r;
    color.green    = g;
    color.blue     = b;
    color.alpha    = a;
    color.pixelate = FALSE;
    axoverlay_set_palette_color(index, &color, &error);
    if (error != NULL) {
        g_error_free(error);
        return FALSE;
    }

    return TRUE;
}

/***** Callback functions ****************************************************/

/**
 * brief A callback function called when an overlay needs adjustments.
 *
 * This function is called to let developers make adjustments to
 * the size and position of their overlays for each stream. This callback
 * function is called prior to rendering every time when an overlay
 * is rendered on a stream, which is useful if the resolution has been
 * updated or rotation has changed.
 *
 * param id Overlay id.
 * param stream Information about the rendered stream.
 * param postype The position type.
 * param overlay_x The x coordinate of the overlay.
 * param overlay_y The y coordinate of the overlay.
 * param overlay_width Overlay width.
 * param overlay_height Overlay height.
 * param user_data Optional user data associated with this overlay.
 */
static void adjustment_cb(gint id,
                          struct axoverlay_stream_data* stream,
                          enum axoverlay_position_type* postype,
                          gfloat* overlay_x,
                          gfloat* overlay_y,
                          gint* overlay_width,
                          gint* overlay_height,
                          gpointer user_data) {
    /* Silence compiler warnings for unused parameters/arguments */
    (void)id;
    (void)postype;
    (void)overlay_x;
    (void)overlay_y;
    (void)user_data;

    // syslog(LOG_INFO, "Adjust callback for overlay: %i x %i", *overlay_width, *overlay_height);
    // syslog(LOG_INFO, "Adjust callback for stream: %i x %i", stream->width, stream->height);

    *overlay_width  = stream->width;
    *overlay_height = stream->height;
}

/**
 * brief A callback function called when an overlay needs to be drawn.
 *
 * This function is called whenever the system redraws an overlay. This can
 * happen in two cases, axoverlay_redraw() is called or a new stream is
 * started.
 *
 * param rendering_context A pointer to the rendering context.
 * param id Overlay id.
 * param stream Information about the rendered stream.
 * param postype The position type.
 * param overlay_x The x coordinate of the overlay.
 * param overlay_y The y coordinate of the overlay.
 * param overlay_width Overlay width.
 * param overlay_height Overlay height.
 * param user_data Optional user data associated with this overlay.
 */
static void render_overlay_cb(gpointer rendering_context,
                              gint id,
                              struct axoverlay_stream_data* stream,
                              enum axoverlay_position_type postype,
                              gfloat overlay_x,
                              gfloat overlay_y,
                              gint overlay_width,
                              gint overlay_height,
                              gpointer user_data) {
    /* Silence compiler warnings for unused parameters/arguments */
    (void)postype;
    (void)user_data;
    (void)overlay_x;
    (void)overlay_y;
    (void)overlay_height;
    (void)overlay_width;

    // gdouble val = FALSE;

    stream_width  = stream->width;   // TODO: SUPERHACKY TO SET THESE HERE
    stream_height = stream->height;  // TODO: SUPERHACKY TO SET THESE HERE
    syslog(LOG_INFO,
           "Setting stream width/height to %dx%d (from %dx%d)",
           stream_width,
           stream_height,
           stream->width,
           stream->height);

    // syslog(LOG_INFO, "Render callback for camera: %i", stream->camera);
    // syslog(LOG_INFO, "Render callback for overlay: %i x %i", overlay_width, overlay_height);
    // syslog(LOG_INFO, "Render callback for stream: %i x %i", stream->width, stream->height);

    for (size_t i = 0; i < object_overlays_length; i++) {
        ObjectOverlay* overlay = &object_overlays[i];

        if (id == overlay->bounding_box_id) {
            draw_rectangle(rendering_context,
                           overlay->left,
                           overlay->top,
                           overlay->right,
                           overlay->bottom,
                           1,
                           5);
        } else if (id == overlay->text_id) {
            draw_text(rendering_context,
                      g_strdup_printf("%s (%f)", overlay->class, overlay->score),
                      overlay->left + ((overlay->right - overlay->left) / 2),
                      overlay->top + ((overlay->bottom - overlay->top) / 2));
        }
    }
}

/**
 * brief Callback function which is called when animation timer has elapsed.
 *
 * This function is called when the animation timer has elapsed, which will
 * update the counter, colors and also trigger a redraw of the overlay.
 *
 * param user_data Optional callback user data.
 */
static gboolean update_overlay_cb(gpointer user_data) {
    /* Silence compiler warnings for unused parameters/arguments */
    (void)user_data;

    GError* error = NULL;

    // Countdown
    counter = counter < 1 ? 10 : counter - 1;

    // if (counter == 0) {
    //     // A small color surprise
    //     top_color    = top_color > 2 ? 1 : top_color + 1;
    //     bottom_color = bottom_color > 2 ? 1 : bottom_color + 1;
    // }

    // Request a redraw of the overlay
    axoverlay_redraw(&error);
    if (error != NULL) {
        /*
         * If redraw fails then it is likely due to that overlayd has
         * crashed. Don't exit instead wait for overlayd to restart and
         * for axoverlay to restore the connection.
         */
        syslog(LOG_ERR, "Failed to redraw overlay (%d): %s", error->code, error->message);
        g_error_free(error);
    }

    return G_SOURCE_CONTINUE;
}
#endif
// ------------------------------------------------------------------------------------------------------------------------

/**
 * @brief Free up resources held by an array of labels.
 *
 * @param labels An array of label string pointers.
 * @param labelFileBuffer Heap buffer containing the actual string data.
 */
static void freeLabels(char** labelsArray, char* labelFileBuffer) {
    free(labelsArray);
    free(labelFileBuffer);
}

/**
 * @brief Reads a file of labels into an array.
 *
 * An array filled by this function should be freed using freeLabels.
 *
 * @param labelsPtr Pointer to a string array.
 * @param labelFileBuffer Pointer to the labels file contents.
 * @param labelsPath String containing the path to the labels file to be read.
 * @param numLabelsPtr Pointer to number which will store number of labels read.
 * @return False if any errors occur, otherwise true.
 */
static bool parseLabels(char*** labelsPtr,
                        char** labelFileBuffer,
                        const char* labelsPath,
                        size_t* numLabelsPtr) {
    // We cut off every row at 60 characters.
    const size_t LINE_MAX_LEN = 60;
    bool ret                  = false;
    char* labelsData          = NULL;  // Buffer containing the label file contents.
    char** labelArray         = NULL;  // Pointers to each line in the labels text.

    struct stat fileStats = {0};
    if (stat(labelsPath, &fileStats) < 0) {
        syslog(LOG_ERR,
               "%s: Unable to get stats for label file %s: %s",
               __func__,
               labelsPath,
               strerror(errno));
        return false;
    }

    // Sanity checking on the file size - we use size_t to keep track of file
    // size and to iterate over the contents. off_t is signed and 32-bit or
    // 64-bit depending on architecture. We just check toward 10 MByte as we
    // will not encounter larger label files and both off_t and size_t should be
    // able to represent 10 megabytes on both 32-bit and 64-bit systems.
    if (fileStats.st_size > (10 * 1024 * 1024)) {
        syslog(LOG_ERR, "%s: failed sanity check on labels file size", __func__);
        return false;
    }

    int labelsFd = open(labelsPath, O_RDONLY);
    if (labelsFd < 0) {
        syslog(LOG_ERR,
               "%s: Could not open labels file %s: %s",
               __func__,
               labelsPath,
               strerror(errno));
        return false;
    }

    size_t labelsFileSize = (size_t)fileStats.st_size;
    // Allocate room for a terminating NULL char after the last line.
    labelsData = malloc(labelsFileSize + 1);
    if (labelsData == NULL) {
        syslog(LOG_ERR, "%s: Failed allocating labels text buffer: %s", __func__, strerror(errno));
        goto end;
    }

    ssize_t numBytesRead  = -1;
    size_t totalBytesRead = 0;
    char* fileReadPtr     = labelsData;
    while (totalBytesRead < labelsFileSize) {
        numBytesRead = read(labelsFd, fileReadPtr, labelsFileSize - totalBytesRead);

        if (numBytesRead < 1) {
            syslog(LOG_ERR, "%s: Failed reading from labels file: %s", __func__, strerror(errno));
            goto end;
        }
        totalBytesRead += (size_t)numBytesRead;
        fileReadPtr += numBytesRead;
    }

    // Now count number of lines in the file - check all bytes except the last
    // one in the file.
    size_t numLines = 0;
    for (size_t i = 0; i < (labelsFileSize - 1); i++) {
        if (labelsData[i] == '\n') {
            numLines++;
        }
    }

    // We assume that there is always a line at the end of the file, possibly
    // terminated by newline char. Either way add this line as well to the
    // counter.
    numLines++;

    labelArray = malloc(numLines * sizeof(char*));
    if (!labelArray) {
        syslog(LOG_ERR, "%s: Unable to allocate labels array: %s", __func__, strerror(errno));
        ret = false;
        goto end;
    }

    size_t labelIdx      = 0;
    labelArray[labelIdx] = labelsData;
    labelIdx++;
    for (size_t i = 0; i < labelsFileSize; i++) {
        if (labelsData[i] == '\n') {
            if (i < (labelsFileSize - 1)) {
                // Register the string start in the list of labels.
                labelArray[labelIdx] = labelsData + i + 1;
                labelIdx++;
            }
            // Replace the newline char with string-ending NULL char.
            labelsData[i] = '\0';
        }
    }

    // Make sure we always have a terminating NULL char after the label file
    // contents.
    labelsData[labelsFileSize] = '\0';

    // Now go through the list of strings and cap if strings too long.
    for (size_t i = 0; i < numLines; i++) {
        size_t stringLen = strnlen(labelArray[i], LINE_MAX_LEN);
        if (stringLen >= LINE_MAX_LEN) {
            // Just insert capping NULL terminator to limit the string len.
            *(labelArray[i] + LINE_MAX_LEN + 1) = '\0';
        }
    }

    *labelsPtr       = labelArray;
    *numLabelsPtr    = numLines;
    *labelFileBuffer = labelsData;

    ret = true;

end:
    if (!ret) {
        freeLabels(labelArray, labelsData);
    }
    close(labelsFd);

    return ret;
}

/**
 * @brief Creates a temporary fd and truncated to correct size and mapped.
 *
 * This convenience function creates temp files to be used for input and output.
 *
 * @param fileName Pattern for how the temp file will be named in file system.
 * @param fileSize How much space needed to be allocated (truncated) in fd.
 * @param mappedAddr Pointer to the address of the fd mapped for this process.
 * @param Pointer to the generated fd.
 * @return Positive errno style return code (zero means success).
 */
static bool createAndMapTmpFile(char* fileName, size_t fileSize, void** mappedAddr, int* convFd) {
    syslog(LOG_INFO,
           "%s: Setting up a temp fd with pattern %s and size %zu",
           __func__,
           fileName,
           fileSize);

    int fd = mkstemp(fileName);
    if (fd < 0) {
        syslog(LOG_ERR, "%s: Unable to open temp file %s: %s", __func__, fileName, strerror(errno));
        goto error;
    }

    // Allocate enough space in for the fd.
    if (ftruncate(fd, (off_t)fileSize) < 0) {
        syslog(LOG_ERR,
               "%s: Unable to truncate temp file %s: %s",
               __func__,
               fileName,
               strerror(errno));
        goto error;
    }

    // Remove since we don't actually care about writing to the file system.
    if (unlink(fileName)) {
        syslog(LOG_ERR,
               "%s: Unable to unlink from temp file %s: %s",
               __func__,
               fileName,
               strerror(errno));
        goto error;
    }

    // Get an address to fd's memory for this process's memory space.
    void* data = mmap(NULL, fileSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    if (data == MAP_FAILED) {
        syslog(LOG_ERR, "%s: Unable to mmap temp file %s: %s", __func__, fileName, strerror(errno));
        goto error;
    }

    *mappedAddr = data;
    *convFd     = fd;

    return true;

error:
    if (fd >= 0) {
        close(fd);
    }

    return false;
}

/**
 * @brief Sets up and configures a connection to larod, and loads a model.
 *
 * Opens a connection to larod, which is tied to larodConn. After opening a
 * larod connection the chip specified by chipString is set for the
 * connection. Then the model file specified by larodModelFd is loaded to the
 * chip, and a corresponding larodModel object is tied to model.
 *
 * @param chipString Speficier for which larod chip to use.
 * @param larodModelFd Fd for a model file to load.
 * @param larodConn Pointer to a larod connection to be opened.
 * @param model Pointer to a larodModel to be obtained.
 * @return false if error has occurred, otherwise true.
 */
static bool setupLarod(const char* chipString,
                       const int larodModelFd,
                       larodConnection** larodConn,
                       larodModel** model) {
    larodError* error       = NULL;
    larodConnection* conn   = NULL;
    larodModel* loadedModel = NULL;
    bool ret                = false;

    // Set up larod connection.
    if (!larodConnect(&conn, &error)) {
        syslog(LOG_ERR, "%s: Could not connect to larod: %s", __func__, error->msg);
        goto end;
    }

    // List available chip id:s
    size_t numDevices = 0;
    syslog(LOG_INFO, "Available chip IDs:");
    const larodDevice** devices;
    devices = larodListDevices(conn, &numDevices, &error);
    for (size_t i = 0; i < numDevices; ++i) {
        syslog(LOG_INFO, "%s: %s", "Chip", larodGetDeviceName(devices[i], &error));
        ;
    }
    const larodDevice* dev = larodGetDevice(conn, chipString, 0, &error);
    loadedModel            = larodLoadModel(conn,
                                 larodModelFd,
                                 dev,
                                 LAROD_ACCESS_PRIVATE,
                                 "object_detection",
                                 NULL,
                                 &error);
    if (!loadedModel) {
        syslog(LOG_ERR, "%s: Unable to load model: %s", __func__, error->msg);
        goto error;
    }
    *larodConn = conn;
    *model     = loadedModel;

    ret = true;

    goto end;

error:
    if (conn) {
        larodDisconnect(&conn, NULL);
    }

end:
    if (error) {
        larodClearError(&error);
    }

    return ret;
}

static gboolean detect_objects(void) {
    struct timeval startTs, endTs;
    unsigned int elapsedMs = 0;

    syslog(LOG_INFO, "--------------------------------------------");

    // Get latest frame from image pipeline.
    VdoBuffer* buf = getLastFrameBlocking(sdImageProvider);
    if (!buf) {
        syslog(LOG_ERR, "buf empty in provider");
        return FALSE;
    }

    VdoBuffer* buf_hq = getLastFrameBlocking(hdImageProvider);
    if (!buf_hq) {
        syslog(LOG_ERR, "buf empty in provider high resolution");
        return FALSE;
    }

    // Get data from latest frame.
    uint8_t* nv12Data    = (uint8_t*)vdo_buffer_get_data(buf);
    uint8_t* nv12Data_hq = (uint8_t*)vdo_buffer_get_data(buf_hq);

    // Covert image data from NV12 format to interleaved uint8_t RGB format.
    gettimeofday(&startTs, NULL);

    memcpy(ppInputAddr, nv12Data, yuyvBufferSize);
    if (!larodRunJob(conn, ppReq, &error)) {
        syslog(LOG_ERR, "Unable to run job to preprocess model: %s (%d)", error->msg, error->code);
        return FALSE;
    }
    memcpy(ppInputAddrHD, nv12Data_hq, widthFrameHD * heightFrameHD * CHANNELS / 2);
    if (!larodRunJob(conn, ppReqHD, &error)) {
        syslog(LOG_ERR, "Unable to run job to preprocess model: %s (%d)", error->msg, error->code);
        return FALSE;
    }

    gettimeofday(&endTs, NULL);

    elapsedMs = (unsigned int)(((endTs.tv_sec - startTs.tv_sec) * 1000) +
                               ((endTs.tv_usec - startTs.tv_usec) / 1000));
    syslog(LOG_INFO, "Converted image in %u ms", elapsedMs);

    // Since larodOutputAddr points to the beginning of the fd we should
    // rewind the file position before each job.
    if (lseek(larodOutput1Fd, 0, SEEK_SET) == -1) {
        syslog(LOG_ERR, "Unable to rewind output file position: %s", strerror(errno));
        return FALSE;
    }

    if (lseek(larodOutput2Fd, 0, SEEK_SET) == -1) {
        syslog(LOG_ERR, "Unable to rewind output file position: %s", strerror(errno));
        return FALSE;
    }

    if (lseek(larodOutput3Fd, 0, SEEK_SET) == -1) {
        syslog(LOG_ERR, "Unable to rewind output file position: %s", strerror(errno));
        return FALSE;
    }

    if (lseek(larodOutput4Fd, 0, SEEK_SET) == -1) {
        syslog(LOG_ERR, "Unable to rewind output file position: %s", strerror(errno));
        return FALSE;
    }

    gettimeofday(&startTs, NULL);
    if (!larodRunJob(conn, infReq, &error)) {
        syslog(LOG_ERR,
               "Unable to run inference on model %s: %s (%d)",
               labelsFile,
               error->msg,
               error->code);
        return FALSE;
    }
    gettimeofday(&endTs, NULL);

    elapsedMs = (unsigned int)(((endTs.tv_sec - startTs.tv_sec) * 1000) +
                               ((endTs.tv_usec - startTs.tv_usec) / 1000));
    syslog(LOG_INFO, "Ran inference for %u ms", elapsedMs);

    float* locations          = (float*)larodOutput1Addr;
    float* classes            = (float*)larodOutput2Addr;
    float* scores             = (float*)larodOutput3Addr;
    float* numberOfDetections = (float*)larodOutput4Addr;

    if ((int)numberOfDetections[0] == 0) {
        syslog(LOG_INFO, "No object is detected");
        return TRUE;
    }

    for (int i = 0; i < numberOfDetections[0]; i++) {
        float top    = locations[4 * i];
        float left   = locations[4 * i + 1];
        float bottom = locations[4 * i + 2];
        float right  = locations[4 * i + 3];

        unsigned int croppedWidthHD = heightFrameHD;

        unsigned int crop_x = left * croppedWidthHD + (widthFrameHD - heightFrameHD) / 2;
        unsigned int crop_y = top * heightFrameHD;
        unsigned int crop_w = (right - left) * croppedWidthHD;
        unsigned int crop_h = (bottom - top) * heightFrameHD;

        if (scores[i] >= threshold / 100.0) {
            syslog(LOG_INFO,
                   "Object %d: Classes: %s - Scores: %f - Locations: [%f,%f,%f,%f]",
                   i,
                   labels[(int)classes[i]],
                   scores[i],
                   top,
                   left,
                   bottom,
                   right);

            unsigned char* crop_buffer = crop_interleaved(ppOutputAddrHD,
                                                          widthFrameHD,
                                                          heightFrameHD,
                                                          CHANNELS,
                                                          crop_x,
                                                          crop_y,
                                                          crop_w,
                                                          crop_h);

            unsigned long jpeg_size    = 0;
            unsigned char* jpeg_buffer = NULL;
            struct jpeg_compress_struct jpeg_conf;
            set_jpeg_configuration(crop_w, crop_h, CHANNELS, quality, &jpeg_conf);
            buffer_to_jpeg(crop_buffer, &jpeg_conf, &jpeg_size, &jpeg_buffer);
            char file_name[32];
            snprintf(file_name, sizeof(char) * 32, "/tmp/detection_%i.jpg", i);
            jpeg_to_file(file_name, jpeg_buffer, jpeg_size);
            free(crop_buffer);
            free(jpeg_buffer);
        }
    }

// ADDED: UPDATE DRAW BOX LOCATION FOR FIRST OBJECT ---------3
#if defined(ENABLE_OVERLAY) || defined(ENABLE_CV25_OVERLAY)
    syslog(LOG_INFO,
           "Desired HDImageHeight/Width %dx%d, height/WidthFrameHd: %dx%d",
           desiredHDImgHeight,
           desiredHDImgWidth,
           heightFrameHD,
           widthFrameHD);

    object_overlays_length = numberOfDetections[0] < OBJECT_OVERLAYS_MAX_LENGTH
                                 ? numberOfDetections[0]
                                 : OBJECT_OVERLAYS_MAX_LENGTH;

    for (size_t i = 0; i < object_overlays_length; i++) {
        ObjectOverlay* overlay = &object_overlays[i];

        float top    = locations[4 * i];
        float left   = locations[4 * i + 1];
        float bottom = locations[4 * i + 2];
        float right  = locations[4 * i + 3];

        get_coordinates(&overlay->top,
                        &overlay->left,
                        &overlay->bottom,
                        &overlay->right,
                        stream_width,
                        stream_height,
                        top,
                        left,
                        bottom,
                        right);

        strcpy(overlay->class, labels[(int)classes[i]]);
        overlay->score = scores[i];
    }

    // float top_1    = locations[4 * 0];
    // float left_1   = locations[4 * 0 + 1];
    // float bottom_1 = locations[4 * 0 + 2];
    // float right_1  = locations[4 * 0 + 3];

    // get_coordinates(&object_top,
    //                 &object_left,
    //                 &object_bottom,
    //                 &object_right,
    //                 stream_width,
    //                 stream_height,
    //                 top_1,
    //                 left_1,
    //                 bottom_1,
    //                 right_1);

    // float top_2    = locations[4 * 1];
    // float left_2   = locations[4 * 1 + 1];
    // float bottom_2 = locations[4 * 1 + 2];
    // float right_2  = locations[4 * 1 + 3];

    // get_coordinates(&object_top_2,
    //                 &object_left_2,
    //                 &object_bottom_2,
    //                 &object_right_2,
    //                 stream_width,
    //                 stream_height,
    //                 top_2,
    //                 left_2,
    //                 bottom_2,
    //                 right_2);

    // syslog(LOG_INFO,
    //        "top1/2: %d %d, left1/2: %d %d, bot1/2: %d %d, right1/2: %d %d",
    //        object_top,
    //        object_top_2,
    //        object_left,
    //        object_left_2,
    //        object_bottom,
    //        object_bottom_2,
    //        object_right,
    //        object_right_2);
#endif

    // Release frame reference to provider.
    returnFrame(sdImageProvider, buf);
    returnFrame(hdImageProvider, buf_hq);

    return TRUE;
}

static gboolean detect_objects_timeout_callback(gpointer user_data) {
    (void)user_data;
    return detect_objects();
}

// BEGIN BBOX
#ifdef ENABLE_CV25_OVERLAY

static void draw_cv25_overlay(void) {
    // ALTERNATIVE 1 ------------
    bbox_destroy(overlay);
    overlay = bbox_view_new(1u);
    if (!overlay) {
        syslog(LOG_INFO, "Failed creating bbox: %s", strerror(errno));
    }

    // If camera lacks video output, this call will succeed but not do anything.
    if (!bbox_video_output(overlay, true)) {
        syslog(LOG_INFO, "Failed enabling video-output for bbox: %s", strerror(errno));
    }
    // END ALTERNATIVE 1 ---------------------

    // BEGIN ALTERNATIVE2
    /*
    bbox_clear(overlay);
    if (!bbox_commit(overlay, 0u)) {
        syslog(LOG_INFO, "Failed to clear bounding boxes: %s", strerror(errno));
    }
    */
    // END ALTERNATIVE2 -------------------------

    bbox_thickness_medium(overlay);

    for (size_t i = 0; i < object_overlays_length; i++) {
        ObjectOverlay* object = &object_overlays[i];

        // Normalize screen coords
        float box_left   = object->left / (float)stream_width;
        float box_top    = object->top / (float)stream_height;
        float box_right  = object->right / (float)stream_width;
        float box_bottom = object->bottom / (float)stream_height;

        // Set outline based on score
        if (object->score >= OVERLAY_SCORE_THRESHOLD) {
            bbox_style_outline(overlay);
        } else {
            // Switch to thick corner style
            bbox_style_corners(overlay);
        }

        // Get color based on label
        bbox_color_t color;
        if (strcmp(object->class, "bed") == 0) {
            color = BOUNDING_BOX_COLOR_GREEN;
        } else if (strcmp(object->class, "chair") == 0) {
            color = BOUNDING_BOX_COLOR_BLUE;
        } else if (strcmp(object->class, "person") == 0) {
            color = BOUNDING_BOX_COLOR_RED;
        } else {
            color = BOUNDING_BOX_COLOR_BLACK;
        }
        bbox_color(overlay, color);

        bbox_rectangle(overlay, box_left, box_top, box_right, box_bottom);
    }

    // Draw bounding boxes
    if (!bbox_commit(overlay, 0u)) {
        syslog(LOG_INFO, "Failed to draw bounding boxes: %s", strerror(errno));
    }
}

static gboolean draw_cv25_overlay_timeout_callback(gpointer user_data) {
    (void)user_data;
    syslog(LOG_INFO, "Draw cv25 callback");

    draw_cv25_overlay();

    return TRUE;
}

#endif
// END BBOX

/**
 * @brief Main function that starts a stream with different options.
 */
int main(int argc, char** argv) {
    // Open the syslog to report messages for "object_detection"
    openlog("object_detection", LOG_PID | LOG_CONS, LOG_USER);

    args_t args;
    if (!parseArgs(argc, argv, &args)) {
        syslog(LOG_ERR, "%s: Could not parse arguments", __func__);
        goto earlyend;
    }

    chipString         = args.chip;
    modelFile          = args.modelFile;
    labelsFile         = args.labelsFile;
    inputWidth         = args.width;
    inputHeight        = args.height;
    desiredHDImgWidth  = 1280;  // args.raw_width;
    desiredHDImgHeight = 720;   // args.raw_height;
    threshold          = 0;     // TODO: Removed: args.threshold;
    quality            = args.quality;

    syslog(LOG_INFO,
           "Input width/height: %dx%d Desired img width/height %dx%d",
           inputWidth,
           inputHeight,
           desiredHDImgWidth,
           desiredHDImgHeight);

    syslog(LOG_INFO, "Finding best resolution to use as model input");
    unsigned int streamWidth  = 0;
    unsigned int streamHeight = 0;
    if (!chooseStreamResolution(inputWidth, inputHeight, &streamWidth, &streamHeight)) {
        syslog(LOG_ERR, "%s: Failed choosing stream resolution", __func__);
        goto end;
    }

    syslog(LOG_INFO,
           "Creating VDO image provider and creating stream %d x %d",
           streamWidth,
           streamHeight);
    sdImageProvider = createImgProvider(streamWidth, streamHeight, 2, VDO_FORMAT_YUV);
    if (!sdImageProvider) {
        syslog(LOG_ERR, "%s: Could not create image provider", __func__);
        goto end;
    }
    syslog(LOG_INFO, "Find the best resolution to save the high resolution image");

    if (!chooseStreamResolution(desiredHDImgWidth,
                                desiredHDImgHeight,
                                &widthFrameHD,
                                &heightFrameHD)) {
        syslog(LOG_ERR, "%s: Failed choosing HD resolution", __func__);
        goto end;
    }
    syslog(LOG_INFO,
           "Creating VDO High resolution image provider and stream %d x %d",
           widthFrameHD,
           heightFrameHD);
    hdImageProvider = createImgProvider(widthFrameHD, heightFrameHD, 2, VDO_FORMAT_YUV);
    if (!hdImageProvider) {
        syslog(LOG_ERR, "%s: Could not create high resolution image provider", __func__);
    }

    // Calculate crop image
    // 1. The crop area shall fill the input image either horizontally or
    //    vertically.
    // 2. The crop area shall have the same aspect ratio as the output image.
    syslog(LOG_INFO, "Calculate crop image");
    float destWHratio = (float)inputWidth / (float)inputHeight;
    float cropW       = (float)streamWidth;
    float cropH       = cropW / destWHratio;
    if (cropH > (float)streamHeight) {
        cropH = (float)streamHeight;
        cropW = cropH * destWHratio;
    }
    unsigned int clipW = (unsigned int)cropW;
    unsigned int clipH = (unsigned int)cropH;
    unsigned int clipX = (streamWidth - clipW) / 2;
    unsigned int clipY = (streamHeight - clipH) / 2;
    syslog(LOG_INFO, "Crop VDO image X=%d Y=%d (%d x %d)", clipX, clipY, clipW, clipH);

    // Create preprocessing maps
    syslog(LOG_INFO, "Create preprocessing maps");
    ppMap = larodCreateMap(&error);
    if (!ppMap) {
        syslog(LOG_ERR, "Could not create preprocessing larodMap %s", error->msg);
        goto end;
    }
    if (!larodMapSetStr(ppMap, "image.input.format", "nv12", &error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", error->msg);
        goto end;
    }
    if (!larodMapSetIntArr2(ppMap, "image.input.size", streamWidth, streamHeight, &error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", error->msg);
        goto end;
    }
    if (!larodMapSetStr(ppMap, "image.output.format", "rgb-interleaved", &error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", error->msg);
        goto end;
    }

    if (!larodMapSetIntArr2(ppMap, "image.output.size", inputWidth, inputHeight, &error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", error->msg);
        goto end;
    }
    ppMapHD = larodCreateMap(&error);
    if (!ppMapHD) {
        syslog(LOG_ERR, "Could not create preprocessing high resolution larodMap %s", error->msg);
        goto end;
    }
    if (!larodMapSetStr(ppMapHD, "image.input.format", "nv12", &error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", error->msg);
        goto end;
    }
    if (!larodMapSetIntArr2(ppMapHD, "image.input.size", widthFrameHD, heightFrameHD, &error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", error->msg);
        goto end;
    }
    if (!larodMapSetStr(ppMapHD, "image.output.format", "rgb-interleaved", &error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", error->msg);
        goto end;
    }
    if (!larodMapSetIntArr2(ppMapHD, "image.output.size", widthFrameHD, heightFrameHD, &error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", error->msg);
        goto end;
    }

    cropMap = larodCreateMap(&error);
    if (!cropMap) {
        syslog(LOG_ERR, "Could not create preprocessing crop larodMap %s", error->msg);
        goto end;
    }
    if (!larodMapSetIntArr4(cropMap, "image.input.crop", clipX, clipY, clipW, clipH, &error)) {
        syslog(LOG_ERR, "Failed setting preprocessing parameters: %s", error->msg);
        goto end;
    }

    // Create larod models
    syslog(LOG_INFO, "Create larod models");
    larodModelFd = open(modelFile, O_RDONLY);
    if (larodModelFd < 0) {
        syslog(LOG_ERR, "Unable to open model file %s: %s", modelFile, strerror(errno));
        goto end;
    }

    syslog(LOG_INFO,
           "Setting up larod connection with chip %s, model %s and label file %s",
           chipString,
           modelFile,
           labelsFile);
    if (!setupLarod(chipString, larodModelFd, &conn, &model)) {
        goto end;
    }

    // Use libyuv as image preprocessing backend
    const char* larodLibyuvPP = "cpu-proc";
    const larodDevice* dev_pp;
    dev_pp  = larodGetDevice(conn, larodLibyuvPP, 0, &error);
    ppModel = larodLoadModel(conn, -1, dev_pp, LAROD_ACCESS_PRIVATE, "", ppMap, &error);
    if (!ppModel) {
        syslog(LOG_ERR,
               "Unable to load preprocessing model with chip %s: %s",
               larodLibyuvPP,
               error->msg);
        goto end;
    } else {
        syslog(LOG_INFO, "Loading preprocessing model with chip %s", larodLibyuvPP);
    }

    // run image processing also on the high resolution frame

    const larodDevice* dev_pp_hd;
    dev_pp_hd = larodGetDevice(conn, larodLibyuvPP, 0, &error);
    ppModelHD = larodLoadModel(conn, -1, dev_pp_hd, LAROD_ACCESS_PRIVATE, "", ppMapHD, &error);
    if (!ppModelHD) {
        syslog(LOG_ERR,
               "Unable to load preprocessing model with chip %s: %s",
               larodLibyuvPP,
               error->msg);
        goto end;
    } else {
        syslog(LOG_INFO, "Loading preprocessing model with chip %s", larodLibyuvPP);
    }

    // Create input/output tensors
    syslog(LOG_INFO, "Create input/output tensors");
    ppInputTensors = larodCreateModelInputs(ppModel, &ppNumInputs, &error);
    if (!ppInputTensors) {
        syslog(LOG_ERR, "Failed retrieving input tensors: %s", error->msg);
        goto end;
    }
    ppOutputTensors = larodCreateModelOutputs(ppModel, &ppNumOutputs, &error);
    if (!ppOutputTensors) {
        syslog(LOG_ERR, "Failed retrieving output tensors: %s", error->msg);
        goto end;
    }

    ppInputTensorsHD = larodCreateModelInputs(ppModelHD, &ppNumInputsHD, &error);
    if (!ppInputTensorsHD) {
        syslog(LOG_ERR, "Failed retrieving input tensors: %s", error->msg);
        goto end;
    }
    ppOutputTensorsHD = larodCreateModelOutputs(ppModelHD, &ppNumOutputsHD, &error);
    if (!ppOutputTensorsHD) {
        syslog(LOG_ERR, "Failed retrieving output tensors: %s", error->msg);
        goto end;
    }

    inputTensors = larodCreateModelInputs(model, &numInputs, &error);
    if (!inputTensors) {
        syslog(LOG_ERR, "Failed retrieving input tensors: %s", error->msg);
        goto end;
    }

    outputTensors = larodCreateModelOutputs(model, &numOutputs, &error);
    if (!outputTensors) {
        syslog(LOG_ERR, "Failed retrieving output tensors: %s", error->msg);
        goto end;
    }

    // Determine tensor buffer sizes
    syslog(LOG_INFO, "Determine tensor buffer sizes");
    const larodTensorPitches* ppInputPitches = larodGetTensorPitches(ppInputTensors[0], &error);
    if (!ppInputPitches) {
        syslog(LOG_ERR, "Could not get pitches of tensor: %s", error->msg);
        goto end;
    }
    yuyvBufferSize                            = ppInputPitches->pitches[0];
    const larodTensorPitches* ppOutputPitches = larodGetTensorPitches(ppOutputTensors[0], &error);
    if (!ppOutputPitches) {
        syslog(LOG_ERR, "Could not get pitches of tensor: %s", error->msg);
        goto end;
    }
    size_t rgbBufferSize = ppOutputPitches->pitches[0];
    size_t expectedSize  = inputWidth * inputHeight * CHANNELS;
    if (expectedSize != rgbBufferSize) {
        syslog(LOG_ERR, "Expected video output size %zu, actual %zu", expectedSize, rgbBufferSize);
        goto end;
    }
    const larodTensorPitches* outputPitches = larodGetTensorPitches(outputTensors[0], &error);
    if (!outputPitches) {
        syslog(LOG_ERR, "Could not get pitches of tensor: %s", error->msg);
        goto end;
    }

    // Allocate space for input tensor
    syslog(LOG_INFO, "Allocate memory for input/output buffers");
    if (!createAndMapTmpFile(PP_SD_INPUT_FILE_PATTERN, yuyvBufferSize, &ppInputAddr, &ppInputFd)) {
        goto end;
    }
    if (!createAndMapTmpFile(OBJECT_DETECTOR_INPUT_FILE_PATTERN,
                             inputWidth * inputHeight * CHANNELS,
                             &larodInputAddr,
                             &larodInputFd)) {
        goto end;
    }
    if (!createAndMapTmpFile(PP_HD_INPUT_FILE_PATTERN,
                             widthFrameHD * heightFrameHD * CHANNELS / 2,
                             &ppInputAddrHD,
                             &ppInputFdHD)) {
        goto end;
    }
    if (!createAndMapTmpFile(PP_HD_OUTPUT_FILE_PATTERN,
                             widthFrameHD * heightFrameHD * CHANNELS,
                             &ppOutputAddrHD,
                             &ppOutputFdHD)) {
        goto end;
    }

    if (!createAndMapTmpFile(OBJECT_DETECTOR_OUT1_FILE_PATTERN,
                             TENSOR1SIZE,
                             &larodOutput1Addr,
                             &larodOutput1Fd)) {
        goto end;
    }
    if (!createAndMapTmpFile(OBJECT_DETECTOR_OUT2_FILE_PATTERN,
                             TENSOR2SIZE,
                             &larodOutput2Addr,
                             &larodOutput2Fd)) {
        goto end;
    }

    if (!createAndMapTmpFile(OBJECT_DETECTOR_OUT3_FILE_PATTERN,
                             TENSOR3SIZE,
                             &larodOutput3Addr,
                             &larodOutput3Fd)) {
        goto end;
    }

    if (!createAndMapTmpFile(OBJECT_DETECTOR_OUT4_FILE_PATTERN,
                             TENSOR4SIZE,
                             &larodOutput4Addr,
                             &larodOutput4Fd)) {
        goto end;
    }

    // Connect tensors to file descriptors
    syslog(LOG_INFO, "Connect tensors to file descriptors");
    syslog(LOG_INFO, "Set pp input tensors");
    if (!larodSetTensorFd(ppInputTensors[0], ppInputFd, &error)) {
        syslog(LOG_ERR, "Failed setting input tensor fd: %s", error->msg);
        goto end;
    }
    if (!larodSetTensorFd(ppOutputTensors[0], larodInputFd, &error)) {
        syslog(LOG_ERR, "Failed setting input tensor fd: %s", error->msg);
        goto end;
    }
    syslog(LOG_INFO, "Set pp input tensors for high resolution frame");
    if (!larodSetTensorFd(ppInputTensorsHD[0], ppInputFdHD, &error)) {
        syslog(LOG_ERR, "Failed setting input tensor fd: %s", error->msg);
        goto end;
    }
    if (!larodSetTensorFd(ppOutputTensorsHD[0], ppOutputFdHD, &error)) {
        syslog(LOG_ERR, "Failed setting input tensor fd: %s", error->msg);
        goto end;
    }

    syslog(LOG_INFO, "Set input tensors");
    if (!larodSetTensorFd(inputTensors[0], larodInputFd, &error)) {
        syslog(LOG_ERR, "Failed setting input tensor fd: %s", error->msg);
        goto end;
    }

    syslog(LOG_INFO, "Set output tensors");
    if (!larodSetTensorFd(outputTensors[0], larodOutput1Fd, &error)) {
        syslog(LOG_ERR, "Failed setting output tensor fd: %s", error->msg);
        goto end;
    }

    if (!larodSetTensorFd(outputTensors[1], larodOutput2Fd, &error)) {
        syslog(LOG_ERR, "Failed setting output tensor fd: %s", error->msg);
        goto end;
    }

    if (!larodSetTensorFd(outputTensors[2], larodOutput3Fd, &error)) {
        syslog(LOG_ERR, "Failed setting output tensor fd: %s", error->msg);
        goto end;
    }

    if (!larodSetTensorFd(outputTensors[3], larodOutput4Fd, &error)) {
        syslog(LOG_ERR, "Failed setting output tensor fd: %s", error->msg);
        goto end;
    }

    // Create job requests
    syslog(LOG_INFO, "Create job requests");
    ppReq = larodCreateJobRequest(ppModel,
                                  ppInputTensors,
                                  ppNumInputs,
                                  ppOutputTensors,
                                  ppNumOutputs,
                                  cropMap,
                                  &error);
    if (!ppReq) {
        syslog(LOG_ERR, "Failed creating preprocessing job request: %s", error->msg);
        goto end;
    }
    ppReqHD = larodCreateJobRequest(ppModelHD,
                                    ppInputTensorsHD,
                                    ppNumInputsHD,
                                    ppOutputTensorsHD,
                                    ppNumOutputsHD,
                                    NULL,
                                    &error);
    if (!ppReqHD) {
        syslog(LOG_ERR,
               "Failed creating high resolution preprocessing job request: %s",
               error->msg);
        goto end;
    }

    // App supports only one input/output tensor.
    infReq = larodCreateJobRequest(model,
                                   inputTensors,
                                   numInputs,
                                   outputTensors,
                                   numOutputs,
                                   NULL,
                                   &error);
    if (!infReq) {
        syslog(LOG_ERR, "Failed creating inference request: %s", error->msg);
        goto end;
    }

    if (labelsFile) {
        if (!parseLabels(&labels, &labelFileData, labelsFile, &numLabels)) {
            syslog(LOG_ERR, "Failed creating parsing labels file");
            goto end;
        }
    }

    syslog(LOG_INFO, "Found %zu input tensors and %zu output tensors", numInputs, numOutputs);
    syslog(LOG_INFO, "Start fetching video frames from VDO");
    if (!startFrameFetch(sdImageProvider)) {
        syslog(LOG_ERR, "Stuck in provider");
        goto end;
    }

    if (!startFrameFetch(hdImageProvider)) {
        syslog(LOG_ERR, "Stuck in provider high resolution");
        goto end;
    }

    // BEGIN INIT BBOX
#ifdef ENABLE_CV25_OVERLAY
    overlay = bbox_view_new(1u);
    if (!overlay) {
        syslog(LOG_INFO, "Failed creating bbox: %s", strerror(errno));
    }

    // If camera lacks video output, this call will succeed but not do anything.
    if (!bbox_video_output(overlay, true)) {
        syslog(LOG_INFO, "Failed enabling video-output for bbox: %s", strerror(errno));
    }

    BOUNDING_BOX_COLOR_RED   = bbox_color_from_rgb(0xff, 0x0, 0x0);
    BOUNDING_BOX_COLOR_GREEN = bbox_color_from_rgb(0x0, 0xff, 0x0);
    BOUNDING_BOX_COLOR_BLUE  = bbox_color_from_rgb(0x0, 0x0, 0xff);
    BOUNDING_BOX_COLOR_BLACK = bbox_color_from_rgb(0xff, 0xff, 0xff);
#endif
// END INIT BBOX

// BEGIN INIT AXOVERLAY ------------------------------
#ifdef ENABLE_OVERLAY
    GError* overlay_error = NULL;
    GError* error_text    = NULL;

    //  Initialize the library
    struct axoverlay_settings settings;
    axoverlay_init_axoverlay_settings(&settings);
    settings.render_callback     = render_overlay_cb;
    settings.adjustment_callback = adjustment_cb;
    settings.select_callback     = NULL;
    settings.backend             = AXOVERLAY_CAIRO_IMAGE_BACKEND;
    axoverlay_init(&settings, &overlay_error);
    if (overlay_error != NULL) {
        syslog(LOG_ERR, "Failed to initialize axoverlay: %s", overlay_error->message);
        g_error_free(overlay_error);
        return 1;
    }

    //  Setup colors
    if (!setup_palette_color(0, 0, 0, 0, 0) || !setup_palette_color(1, 255, 0, 0, 255) ||
        !setup_palette_color(2, 0, 255, 0, 255) || !setup_palette_color(3, 0, 0, 255, 255)) {
        syslog(LOG_ERR, "Failed to setup palette colors");
        return 1;
    }

    // Get max resolution for width and height
    gint camera_width = axoverlay_get_max_resolution_width(1, &overlay_error);
    if (overlay_error != NULL) {
        syslog(LOG_ERR, "Failed to get max resolution width: %s", overlay_error->message);
        g_error_free(overlay_error);
        overlay_error = NULL;
    }

    gint camera_height = axoverlay_get_max_resolution_height(1, &overlay_error);
    if (overlay_error != NULL) {
        syslog(LOG_ERR, "Failed to get max resolution height: %s", overlay_error->message);
        g_error_free(overlay_error);
        overlay_error = NULL;
    }

    syslog(LOG_INFO, "Max resolution (width x height): %i x %i", camera_width, camera_height);

    // Create a large overlay using Palette color space
    // struct axoverlay_overlay_data data;
    // setup_axoverlay_data(&data);
    // data.width      = camera_width;
    // data.height     = camera_height;
    // data.colorspace = AXOVERLAY_COLORSPACE_4BIT_PALETTE;
    // overlay_id      = axoverlay_create_overlay(&data, NULL, &overlay_error);
    // if (overlay_error != NULL) {
    //     syslog(LOG_ERR, "Failed to create first overlay: %s", overlay_error->message);
    //     g_error_free(overlay_error);
    //     return 1;
    // }

    for (int i = 0; i < OBJECT_OVERLAYS_MAX_LENGTH; i++) {
        ObjectOverlay* object = &object_overlays[i];

        // Bounding box
        struct axoverlay_overlay_data* bounding_box = &object->bounding_box;
        setup_axoverlay_data(bounding_box);
        bounding_box->width      = camera_width;
        bounding_box->height     = camera_height;
        bounding_box->colorspace = AXOVERLAY_COLORSPACE_4BIT_PALETTE;
        object->bounding_box_id  = axoverlay_create_overlay(bounding_box, NULL, &overlay_error);
        if (overlay_error != NULL) {
            syslog(LOG_ERR, "Failed to create first overlay: %s", overlay_error->message);
            g_error_free(overlay_error);
            return 1;
        }

        // Text
        struct axoverlay_overlay_data* text = &object->text;
        setup_axoverlay_data(text);
        text->width      = camera_width;
        text->height     = camera_height;
        text->colorspace = AXOVERLAY_COLORSPACE_ARGB32;
        object->text_id  = axoverlay_create_overlay(text, NULL, &error_text);
        if (error_text != NULL) {
            syslog(LOG_ERR, "Failed to create second overlay: %s", error_text->message);
            g_error_free(error_text);
            return 1;
        }
    }

    // Create an text overlay using ARGB32 color space
    // struct axoverlay_overlay_data data_text;
    // setup_axoverlay_data(&data_text);
    // data_text.width      = camera_width;
    // data_text.height     = camera_height;
    // data_text.colorspace = AXOVERLAY_COLORSPACE_ARGB32;
    // overlay_id_text      = axoverlay_create_overlay(&data_text, NULL, &error_text);
    // if (error_text != NULL) {
    //     syslog(LOG_ERR, "Failed to create second overlay: %s", error_text->message);
    //     g_error_free(error_text);
    //     return 1;
    // }

    // Draw overlays
    axoverlay_redraw(&overlay_error);
    if (overlay_error != NULL) {
        syslog(LOG_ERR, "Failed to draw overlays: %s", overlay_error->message);
        // axoverlay_destroy_overlay(overlay_id, &overlay_error);
        // axoverlay_destroy_overlay(overlay_id_text, &error_text);
        axoverlay_cleanup();
        g_error_free(overlay_error);
        g_error_free(error_text);
        return 1;
    }

    // Start animation timer
    g_timeout_add(SLEEP_PERIOD_MS, update_overlay_cb, NULL);
#endif

#ifdef ENABLE_CV25_OVERLAY
    g_timeout_add(SLEEP_PERIOD_MS, draw_cv25_overlay_timeout_callback, NULL);
#endif

    // END INIT AXOVERLAY ------------------------------

    g_timeout_add(SLEEP_PERIOD_MS, detect_objects_timeout_callback, NULL);

    // Enter main loop
    GMainLoop* main_loop = g_main_loop_new(NULL, FALSE);
    g_main_loop_run(main_loop);

    // TODO: LOOP USED TO BE HERE

    syslog(LOG_INFO, "Stop streaming video from VDO");
    if (!stopFrameFetch(sdImageProvider)) {
        goto end;
    }

    ret = true;

end:
    if (sdImageProvider) {
        destroyImgProvider(sdImageProvider);
    }
    if (hdImageProvider) {
        destroyImgProvider(hdImageProvider);
    }
    // Only the model handle is released here. We count on larod service to
    // release the privately loaded model when the session is disconnected in
    // larodDisconnect().
    larodDestroyMap(&ppMap);
    larodDestroyMap(&cropMap);
    larodDestroyMap(&ppMapHD);
    larodDestroyModel(&ppModel);
    larodDestroyModel(&ppModelHD);
    larodDestroyModel(&model);
    if (conn) {
        larodDisconnect(&conn, NULL);
    }
    if (larodModelFd >= 0) {
        close(larodModelFd);
    }
    if (larodInputAddr != MAP_FAILED) {
        munmap(larodInputAddr, inputWidth * inputHeight * CHANNELS);
    }
    if (larodInputFd >= 0) {
        close(larodInputFd);
    }
    if (ppInputAddr != MAP_FAILED) {
        munmap(ppInputAddr, inputWidth * inputHeight * CHANNELS);
    }
    if (ppInputFd >= 0) {
        close(ppInputFd);
    }
    if (ppInputAddrHD != MAP_FAILED) {
        munmap(ppInputAddrHD, widthFrameHD * heightFrameHD * CHANNELS / 2);
    }
    if (ppInputFdHD >= 0) {
        close(ppInputFdHD);
    }
    if (ppOutputAddrHD != MAP_FAILED) {
        munmap(ppOutputAddrHD, widthFrameHD * heightFrameHD * CHANNELS);
    }
    if (ppOutputFdHD >= 0) {
        close(ppOutputFdHD);
    }
    if (cropAddr != MAP_FAILED) {
        munmap(cropAddr, widthFrameHD * heightFrameHD * CHANNELS);
    }
    if (larodOutput1Addr != MAP_FAILED) {
        munmap(larodOutput1Addr, TENSOR1SIZE);
    }

    if (larodOutput2Addr != MAP_FAILED) {
        munmap(larodOutput2Addr, TENSOR2SIZE);
    }

    if (larodOutput3Addr != MAP_FAILED) {
        munmap(larodOutput3Addr, TENSOR3SIZE);
    }

    if (larodOutput4Addr != MAP_FAILED) {
        munmap(larodOutput4Addr, TENSOR4SIZE);
    }

    if (larodOutput1Fd >= 0) {
        close(larodOutput1Fd);
    }

    if (larodOutput2Fd >= 0) {
        close(larodOutput2Fd);
    }

    if (larodOutput3Fd >= 0) {
        close(larodOutput3Fd);
    }

    if (larodOutput4Fd >= 0) {
        close(larodOutput4Fd);
    }
    larodDestroyJobRequest(&ppReq);
    larodDestroyJobRequest(&ppReqHD);
    larodDestroyJobRequest(&infReq);
    larodDestroyTensors(conn, &inputTensors, numInputs, &error);
    larodDestroyTensors(conn, &outputTensors, numOutputs, &error);
    larodClearError(&error);

    if (labels) {
        freeLabels(labels, labelFileData);
    }

earlyend:
    syslog(LOG_INFO, "Exit %s", argv[0]);
    return ret ? EXIT_SUCCESS : EXIT_FAILURE;
}

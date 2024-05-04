#pragma comment(lib, "k4a.lib")
#include <k4a/k4a.h>
#include <stdio.h>
#include <stdlib.h>
#include <Python.h>
#include <opencv2/opencv.hpp>

int main() {
    k4a_capture_t capture;
    uint2_t count = k4a_device_get_installed_count();
    if (count == 0) {
        printf("No k4a devices attached!\n");
        return 1;
    }

    // Open the first plugged in Kinect device
    k4a_device_t device = NULL;
    if (K4A_FAILED(k4a_device_open(K4A_DEVICE_DEFAULT, &device))) {
        printf("Failed to open k4a device!\n");
        return 1;
    }

    // Get the size of the serial number
    size_t serial_size = 0;
    k4a_device_get_serialnum(device, NULL, &serial_size);

    // Allocate memory for the serial, then acquire it
    char *serial = (char*)(malloc(serial_size));
    k4a_device_get_serialnum(device, serial, &serial_size);
    printf("Opened device: %s\n", serial);
    free(serial);

    // Configure a stream of 4096x3072 BRGA color data at 15 frames per second
    k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    config.camera_fps       = K4A_FRAMES_PER_SECOND_15;
    config.color_format     = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    config.color_resolution = K4A_COLOR_RESOLUTION_3072P;
    //config.color_resolution = K4A_COLOR_RESOLUTION_2160P; //the stack overflow guy had this not sure why it would be a problem
    config.depth_mode = K4A_DEPTH_MODE_WFOV_UNBINNED; // <==== For Depth image // not sure what the point of this is
    
    //create transformation handle to use for making depth->color image
    k4a_calibration_t *calibration;
    k4a_device_get_calibration(device, config.depth_mode, config.color_resolution, calibration);	
    k4a_transformation_t transformationHandle = k4a_transformation_create(calibration);	
    
    // Start the camera with the given configuration
    if (K4A_FAILED(k4a_device_start_cameras(device, &config))) {
        printf("Failed to start cameras!\n");
        k4a_device_close(device);
        return 1;
    }
    // Initialize the Python interpreter
    Py_Initialize();

    // Import the Python module
    PyObject *HandTest = PyImport_ImportModule("HandTest");

    // Get a reference to the Python function
    PyObject *drawHands = PyObject_GetAttrString(HandTest, "drawHands");

    //make a 2d array to hold coordinates from python call
    double **pointerFinger = (double **)malloc(2 * sizeof(double *));
    for (int i = 0; i < 2; i++) {
        pointerFinger[i] = (double *)malloc(2 * sizeof(double));
    }

    bool lookingForHands = true;
    while(lookingForHands) {
        if (k4a_device_get_capture(device, &capture, K4A_WAIT_INFINITE) == K4A_WAIT_RESULT_SUCCEEDED) {
            // get image metadata
            k4a_image_t colorImage = k4a_capture_get_color_image(capture);
            k4a_image_t depthImage = k4a_capture_get_depth_image(capture); 

            //send color image to MediaPipe
            if (colorImage != NULL) {
                // get raw buffer
                uint8_t* buffer = k4a_image_get_buffer(colorImage);

                // convert the raw buffer to cv::Mat
                int rows = k4a_image_get_height_pixels(colorImage);
                int cols = k4a_image_get_width_pixels(colorImage);
                cv::Mat colorMat(rows , cols, CV_8UC4, (void*)buffer, cv::Mat::AUTO_STEP);

                // Serialize the OpenCV Mat
                std::vector<uchar> buf;
                cv::imencode(".jpg", colorMat, buf);
                Py_buffer pybuf = {buf.data(), buf.size()};

                // Prepare cv::Mat to send to drawHands()
                PyObject *curFrame = PyTuple_Pack(1, PyBytes_FromStringAndSize((char*)pybuf.buf, pybuf.len));

                // Call drawHands()
                PyObject *fingerCoords = PyObject_CallObject(drawHands, curFrame);

                //extract c doubles from python tuple
                // Process the result (a list of coordinates)
                Py_ssize_t size = PyList_Size(fingerCoords);
                for (Py_ssize_t i = 0; i < size; ++i) {
                    PyObject *coords = PyList_GetItem(fingerCoords, i);
                    if (PyTuple_Check(coords) && PyTuple_Size(coords) == 3) {
                        //put coordinates in 2d array
                        pointerFinger[i][0] = PyFloat_AsDouble(PyTuple_GetItem(coords, 0));
                        pointerFinger[i][1] = PyFloat_AsDouble(PyTuple_GetItem(coords, 1));
                        //pointerFinger[i][2] = PyFloat_AsDouble(PyTuple_GetItem(coords, 2));
                    }
                }

                //send the values in pointerFinger somewhere to do some stuff

                // Clean up python objects
                Py_DECREF(fingerCoords);
                Py_DECREF(curFrame);
            }
            //get 2d working first
                // //Map depth image to color image
                // if (depthImage != NULL) {
                //     //create transformed image from depth->color
                //     k4a_image_t *transformedDepthImage;
                //     k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16, k4a_image_get_width_pixels(colorImage), k4a_image_get_height_pixels(colorImage),
                //                      sizeof(uint16_t) * k4a_image_get_width_pixels(colorImage), transformedDepthImage);	
                //     k4a_transformation_depth_image_to_color_camera(transformationHandle, depthImage, transformedDepthImage);
                //     int index = y * stride + x * (int)k4a_image_get_bytes_per_pixel(format);
                //     uint16_t depth = *(uint16_t*)(k4a_image_get_buffer(img) + index);
                //     k4a_calibration_2d_to_3d	(const k4a_calibration_t * 	calibration,
                //                                    const k4a_float2_t * 	source_point2d,
                //                                     const float 	source_depth_mm,
                //                                     const k4a_calibration_type_t 	source_camera,
                //                                     const k4a_calibration_type_t 	target_camera,
                //                                     k4a_float3_t * 	target_point3d_mm,
                //                                     int * 	valid 
                //                                     )	
                // }

            k4a_image_release(colorImage);
            //k4a_image_release(depthImage);
        }
    }
    //clean up python module and method
    Py_DECREF(drawHands);
    Py_DECREF(HandTest);

    // Shutdown the Python interpreter
    Py_Finalize();

    //free pointerFinger
    free(pointerFinger[0]);
    free(pointerFinger[1]);
    free(pointerFinger);

    // Shut down the camera when finished with application logic
    k4a_device_stop_cameras(device);
    k4a_device_close(device);

    return 0;
}


//gcc my_program.c -o my_program -I/usr/include/python3.8 -I/path/to/opencv/include -L/path/to/opencv/lib -lpython3.8 -lopencv_core -lopencv_imgcodecs

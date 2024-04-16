#pragma comment(lib, "k4a.lib")
#include <k4a/k4a.h>
#include <stdio.h>
#include <stdlib.h>
#include <Python.h>
#include <opencv2/opencv.hpp>

int main()
{
    k4a_capture_t capture;
    uint32_t count = k4a_device_get_installed_count();
    if (count == 0)
    {
        printf("No k4a devices attached!\n");
        return 1;
    }

    // Open the first plugged in Kinect device
    k4a_device_t device = NULL;
    if (K4A_FAILED(k4a_device_open(K4A_DEVICE_DEFAULT, &device)))
    {
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
    //config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED; // <==== For Depth image // not sure what the point of this is

    
    // Start the camera with the given configuration
    if (K4A_FAILED(k4a_device_start_cameras(device, &config)))
    {
        printf("Failed to start cameras!\n");
        k4a_device_close(device);
        return 1;
    }


    k4a_image_t colorImage = k4a_capture_get_color_image(capture); // get image metadata
    if (colorImage != NULL)
    {
        // you can check the format with this function
        k4a_image_format_t format = k4a_image_get_format(colorImage); // K4A_IMAGE_FORMAT_COLOR_BGRA32 

        // get raw buffer
        uint8_t* buffer = k4a_image_get_buffer(colorImage);

        // convert the raw buffer to cv::Mat
        int rows = k4a_image_get_height_pixels(colorImage);
        int cols = k4a_image_get_width_pixels(colorImage);
        cv::Mat colorMat(rows , cols, CV_8UC4, (void*)buffer, cv::Mat::AUTO_STEP);

        // Initialize the Python interpreter
        Py_Initialize();

        // Import the Python module
        PyObject *pModule = PyImport_ImportModule("HandTest");

        // Get a reference to the Python function
        PyObject *pFunction = PyObject_GetAttrString(pModule, "drawHands");


        // Serialize the OpenCV Mat
        std::vector<uchar> buf;
        cv::imencode(".jpg", colorMat, buf);
        Py_buffer pybuf = {buf.data(), buf.size()};

        // Prepare arguments for the Python function
        PyObject *pArgs = PyTuple_Pack(1, PyBytes_FromStringAndSize((char*)pybuf.buf, pybuf.len));

        // Call the Python function
        PyObject_CallObject(pFunction, pArgs);

        // Clean up
        Py_DECREF(pArgs);
        Py_DECREF(pFunction);
        Py_DECREF(pModule);

        // Shutdown the Python interpreter
        Py_Finalize();

        k4a_image_release(colorImage);
    }

    // Camera capture and application specific code would go here


    // Shut down the camera when finished with application logic
    k4a_device_stop_cameras(device);
    k4a_device_close(device);

    return 0;
}


//gcc my_program.c -o my_program -I/usr/include/python3.8 -I/path/to/opencv/include -L/path/to/opencv/lib -lpython3.8 -lopencv_core -lopencv_imgcodecs

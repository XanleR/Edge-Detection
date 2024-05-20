#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <cstdlib>

void myGradient(const cv::Mat& source, cv::Mat& gradient_out, cv::Mat_<float>& angles) {
    angles = cv::Mat_<float>::zeros(source.size());
    gradient_out = cv::Mat::zeros(source.size(), CV_8UC1);

    auto data = source.data;
    int step = source.step;

    int row_1 = source.rows - 1;
    int col_1 = source.cols - 1;

    int i, j;

    for (i = 1; i < row_1; i++) {
        for (j = 1; j < col_1; j++) {
            uchar pixel_00 = data[(i - 1)*step + j - 1];
            uchar pixel_01 = data[(i - 1)*step + j];
            uchar pixel_02 = data[(i - 1) * step + j + 1];
            uchar pixel_10 = data[(i)*step + j - 1];
            //uchar pixel_11 data[r>(i, j);
            uchar pixel_12 = data[(i)*step + j + 1];
            uchar pixel_20 = data[(i + 1) * step + j - 1];
            uchar pixel_21 = data[(i + 1) * step + j];
            uchar pixel_22 = data[(i + 1) * step + j + 1];


            float der_x = pixel_02 + (2 * pixel_12) + pixel_22 - pixel_00 - (2 * pixel_10) - pixel_20;
            float der_y = pixel_00 + (2 * pixel_01) + pixel_02 - pixel_20 - (2 * pixel_21) - pixel_22;

            float fi = std::atan2(der_y, der_x) * 57.2957795131 + 180 + 22.5f ;

            fi -= fi > 360 ? 0 : 180;
            

            angles.at<float>(i, j) = fi;



            gradient_out.at<uchar>(i, j) = std::sqrtf(der_x * der_x + der_y * der_y);

        }
    }
    

}


void myNonMaximumSupression(cv::Mat& gradient, cv::Mat_<float>& angles) {
    int row_1 = gradient.rows - 1;
    int col_1 = gradient.cols - 1;

    auto angleData = angles.data;
    int angleStep = angles.step;
    float angle;
    uchar previous, next;

    int i, j;
    for (i = 1; i < row_1; i++) {
        for (j = 1; j < col_1; j++) {

            angle = angleData[i*angleStep + j];
            uchar& value = gradient.at<uchar>(i, j);

            /*
            __________
            |00|01|02|
            |10|11|12|
            |20|21|22|
            ----------
            */

            //right down
            if ((0 < angle && angle <= 45) || (angle > 180 && angle <= 225)) {
                next = gradient.at<uchar>(i + 1, j + 1); // 22
                previous = gradient.at<uchar>(i - 1, j - 1); // 00
            }
            //down
            else if ((45 < angle && angle <= 90) || (angle > 225 && angle <= 270)) {
                next = gradient.at<uchar>(i + 1, j); // 21
                previous = gradient.at<uchar>(i - 1, j - 1); // 01
            }
            //left down
            else if ((90 < angle && angle <= 135) || (angle > 270 && angle <= 315)) {
                next = gradient.at<uchar>(i + 1, j - 1); // 20
                previous = gradient.at<uchar>(i - 1, j + 1); // 02
            }
            //horizontal
            else if (angle == 0 || (135 < angle && angle <= 180) || (angle > 315 && angle <= 360)) {
                next = gradient.at<uchar>(i, j - 1); // 10
                previous = gradient.at<uchar>(i, j + 1); // 12
            }

            if (value < next || value < previous) {
                value = 0;
            }
        }
    }
}


void myTreshold(cv::Mat& source, const float treshold) {
    int row = source.rows;
    int col = source.cols;

    auto data = source.data;
    int step = source.step;

    int i, j;
    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            source.at<uchar>(i, j) = data[i * step + j] >= treshold ? 255 : 0;
        }
    }
}

void myOptimizedTreshold(cv::Mat& source, cv::Mat& lowOut, cv::Mat& highOut, const float lowTreshold, const float highTreshold) {
    lowOut = cv::Mat(source.size(), source.type());
    highOut = cv::Mat(source.size(), source.type());
    auto data = source.data;
    int step = source.step;

    int row = source.rows;
    int col = source.cols;

    int i, j;
    for (i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            lowOut.at<uchar>(i, j) = data[i * step + j] >= lowTreshold ? 255 : 0;
            highOut.at<uchar>(i, j) = data[i * step + j] >= highTreshold ? 255 : 0;
        }
    }

}


void myDoubleTreshold(cv::Mat& source, cv::Mat& result, const float lowTreshold, const float highTreshold,
        const int edgeTrackingWindow = 3) {
    cv::Mat strongEdges, weakEdges;

    //myTreshold(strongEdges, highTreshold);
    //myTreshold(weakEdges, lowTreshold);
    myOptimizedTreshold(source, weakEdges, strongEdges, lowTreshold, highTreshold);

    auto strongEdgeData = strongEdges.data;
    int strongEdgeStep = strongEdges.step;

    auto weakEdgeData = weakEdges.data;
    int weakEdgeStep = weakEdges.step;

    int rows = source.rows;
    int cols = source.cols;

    //cv::imshow("highTreshold", strongEdges);
    //cv::imshow("lowTreshold", weakEdges);

    
    int i,j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            if (strongEdgeData[i*strongEdgeStep+j] == 255) {
                source.at<uchar>(i, j) = 255;
                continue;
            }
            if (weakEdgeData[i*weakEdgeStep+j] == 255) {
                //std::cout << "\nWeakEdge";
                source.at<uchar>(i, j) = 100;
                continue;
            }
            source.at<uchar>(i, j) = 0;
        }
    }
    cv::namedWindow("Double Treshold", cv::WINDOW_NORMAL);
    cv::imshow("Double Treshold", result);

    std::queue<cv::Point> edgePoints;

    
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            if (strongEdgeData[i*strongEdgeStep+j] == 255) {
                edgePoints.push(cv::Point(j, i));
            }
        }
    }
    
    
    int difference=1;
    if (edgeTrackingWindow % 2 == 1 && edgeTrackingWindow > 0) {
        difference = (edgeTrackingWindow - 1) / 2;
    }


    int bottom = -difference;
    int top = difference;


    int n, m;
    int row, col;
    while (!edgePoints.empty()) {
        cv::Point current = edgePoints.front();
        edgePoints.pop();

        for (m = bottom; m <= top; m++) {
            for (n = bottom; n <= top; n++) {
                row = current.y + m;
                col = current.x + n;
                if (m == 0 && n == 0 || row < 0 || row >= weakEdges.rows || col < 0 || col >= weakEdges.cols) {
                    continue;
                }
                

                if (strongEdges.at<uchar>(row, col) != 255 && weakEdgeData[row*weakEdgeStep+col] == 255) {  //is gray pixel
                    
                    strongEdges.at<uchar>(row, col) = 255;
                    edgePoints.push(cv::Point(col, row));
                }
            }
        }
    }

    result = strongEdges.clone();

}


void myCanny(const cv::Mat& source, cv::Mat& result, float lowTreshold, float highTreshold, int edgeTrackingWindow = 3) {
    cv::Mat_<float> angles;

    
    myGradient(source, result, angles);

    
    
    
    //cv::imshow("gradient", gradient);
    myNonMaximumSupression(result, angles);
    
    //cv::imshow("DoubleTreshold2", gradient);
    myDoubleTreshold(result, result, lowTreshold, highTreshold, edgeTrackingWindow);
    //cv::namedWindow("DoubleTreshold", cv::WINDOW_NORMAL);
    //cv::imshow("DoubleTreshold", gradient);
}

void printDetails(const int lowTreshold, const int highTreshold, const int changeValue, const int edgeTrackingWindow) {
    std::system("cls");
    std::cout << "\n------------------------------"
        << "\n LowTreshold: " << lowTreshold
        << "\n HighTreshold: " << highTreshold
        << "\n\n ChangeValue: " << changeValue
        << "\n EdgeTrackingWindow: " << edgeTrackingWindow
        << "\n------------------------------"
        << "\n\n Controls"
        << "\n --------"
        << "\n UP_ARROW     = HighTreshold  + " << changeValue
        << "\n DOWN_ARROW   = HighTreshold  - " << changeValue
        << "\n LEFT ARROW   = LowTreshold   + " << changeValue
        << "\n RIGHT ARROW  = LowTreshold   - " << changeValue
        << "\n W            = ChangeValue   + 1"
        << "\n S            = ChangeValue   - 1"
        << "\n D            = EdgeTrackingWindow + 2"
        << "\n A            = EdgeTrackingWindow - 2"
        << "\n\n ESC - Quit";



    std::cout << "\n\n";
}

int main()
{
    std::string imageName = "Lenna_(test_image).png";
    std::string imageFolder = "C:/Users/dani2/Desktop/funky/";

    cv::Mat image = cv::imread(imageFolder + imageName, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "Error: Unable to open image at '" << imageFolder + imageName << "'." << std::endl;
        return -1;
    }

    cv::Mat original = image.clone();

    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(image, image, cv::Size(5, 5), 150);
    //cv::GaussianBlur(image, image, cv::Size(5, 5), 150);


    cv::Mat result;
    float lowTreshold = 30;
    float highTreshold = 70;
    int changeValue = 10;
    int edgeTrackingWindow = 3;

    
    myCanny(image, result, lowTreshold, highTreshold);

    

    cv::namedWindow("Canny", cv::WINDOW_NORMAL);
    cv::imshow("Canny", result);

    cv::namedWindow("Opened Image", cv::WINDOW_NORMAL);
    cv::imshow("Opened Image", original);
    
    

    printDetails(lowTreshold, highTreshold, changeValue, edgeTrackingWindow);


    bool quit = false;

    while (!quit) {
        int key = cv::waitKeyEx(0);

        switch (key)
        {
        case 27: // ESC
            quit = true;
            break;
        case 2490368: // UP ARROW ( HighTreshold + 10 )
            highTreshold = std::min(highTreshold + changeValue, 255.0f);
            break;
        case 2621440: //DOWN ARROW ( HighTreshold -10 )
            highTreshold = std::max(highTreshold - changeValue, lowTreshold + 1);
            break;
        case 2424832: //LEFT ARROW ( LowTreshold -10 )
            lowTreshold = std::max(lowTreshold - changeValue, 0.0f);
            break;
        case 2555904: // RIGHT ARROW ( LowTreshold + 10 )
            lowTreshold = std::min(lowTreshold + changeValue, highTreshold - 1);
            break;
        case 119: // W ( ChangeValue + 1 )
            changeValue = std::min(changeValue + 1, 10);
            break;
        case 115: // S ( ChangValue - 1)
            changeValue = std::max(changeValue - 1, 1);
            break;
        case 100: // D ( EdgeTrackingWindow + 2 )
            edgeTrackingWindow = std::min(edgeTrackingWindow + 2, 127);
            break;
        case 97: // A ( EdgeTrackingWindow - 2 )
            edgeTrackingWindow = std::max(edgeTrackingWindow - 2, 1);
            break;
        default:
            break;
        }

        printDetails(lowTreshold, highTreshold, changeValue, edgeTrackingWindow);

        myCanny(image, result, lowTreshold, highTreshold, edgeTrackingWindow);

        cv::imshow("Canny", result);
        cv::imwrite(imageFolder + "asd.jpg", result);


    }
    

    
    
    
    //cv::imwrite("C:/Users/dani2/Desktop/asdasd.png", result);
    cv::destroyAllWindows();
    

    return 0;
}

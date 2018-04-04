QT       += core gui
QT       += widgets

QMAKE_CXXFLAGS  +=  -Wno-reorder \
                    -Wno-unused-parameter

SOURCES += main.cpp \
        IlluminatedRegion.cpp


HEADERS += \
        IlluminatedRegion.h

INCLUDEPATH += ../../build/opencv/include/opencv2


LIBS    += -L../../build/opencv/lib \
        -lopencv_highgui310 \
        -lopencv_core310 \
        -lopencv_imgproc310 \
        -lopencv_features2d310 \
        -lopencv_imgcodecs310 \
        -lopencv_videoio310 \
        -lopencv_video310

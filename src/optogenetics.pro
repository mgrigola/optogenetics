QT       += core gui widgets


TARGET = optogen    # name of the .exe file
TEMPLATE = app      # project type, equals: application, library, etc. app is default

DEFINES += QT_DEPRECATED_WARNINGS                   # emit warnings if you use any feature of Qt which as been marked as deprecated (
DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

QMAKE_CXXFLAGS  +=  -Wno-reorder \
                    -Wno-unused-parameter

SOURCES += main.cpp \
        MainWindow.cpp \
        IlluminatedRegion.cpp \
    RegionHandler.cpp

HEADERS  += \
        MainWindow.h \
        IlluminatedRegion.h \
    RegionHandler.h

INCLUDEPATH += ../../build/opencv/include


LIBS    += -L../../build/opencv/lib \
        -lopencv_highgui310 \
        -lopencv_core310 \
        -lopencv_imgproc310 \
        -lopencv_features2d310 \
        -lopencv_imgcodecs310 \
        -lopencv_videoio310 \
        -lopencv_video310

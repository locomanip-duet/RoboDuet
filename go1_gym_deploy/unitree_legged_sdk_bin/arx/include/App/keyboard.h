#ifndef _keyboard_H_
#define _keyboard_H_

#include <cmath>
#include <iostream>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <thread>
#include <atomic>

struct key_cmd
{

    bool is_recording = false;
    bool is_playing = false;
    bool is_returning = false; 

};

class arx5_keyboard
{

public:
    arx5_keyboard(void);
    ~arx5_keyboard()=default;

    void detectKeyPress();
    std::atomic<char> keyPress{'\0'};
    key_cmd arx_key_cmd;
};


#endif
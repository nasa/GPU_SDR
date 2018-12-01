#pragma once
#ifndef USRP_PRINT_INCLUDED
#define USRP_PRINT_INCLUDED
#include <iostream>
#include <string.h>
#include <sstream>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
void print_error(std::string text);

void print_warning(std::string text);

void print_debug(std::string text, double value);
#endif

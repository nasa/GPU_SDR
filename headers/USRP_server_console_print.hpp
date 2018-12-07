//! @file USRP_server_console_print.hpp
/* @brief Containd definitions of functions for cool printing on linux console.
 *
 * Avoid early import error.
 * @todo Should be integrated in settings.
*/
#pragma once
//! @endcond
#ifndef USRP_PRINT_INCLUDED
#define USRP_PRINT_INCLUDED
#include <iostream>
#include <string.h>
#include <sstream>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
//! @endcond
void print_error(std::string text);

void print_warning(std::string text);

void print_debug(std::string text, double value);
#endif

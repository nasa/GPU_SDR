#pragma once
#ifndef USRP_JSON_INCLUDED
#define USRP_JSON_INCLUDED
#include "USRP_server_settings.hpp"
#include <boost/property_tree/json_parser.hpp>

#define MAX_MSG_LEN = 10000
//this function will read the arrays inside a json file and put them in a std vector.
template <typename T>
std::vector<T> as_vector(boost::property_tree::ptree const& pt,boost::property_tree::ptree::key_type const& key,boost::property_tree::ptree::key_type const& sub_key = "NULL");

//convert a json string into a parameter object
bool string2param(std::string data, usrp_param &my_parameter);

bool is_pfb_active(param ant_parameter);

//check if the parameters are physically viable
bool chk_param(usrp_param *parameter);

std::string server_ack(std::string payload);

std::string server_nack(std::string payload);

#endif

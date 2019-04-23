#include "USRP_file_writer.hpp"

using namespace H5;

//the initialization needs a rx queue to get packets and a memory to dispose of them
//NOTE: the file writer should always be the last element of the chain unless using a very fast storage support
H5_file_writer::H5_file_writer(rx_queue* init_queue, preallocator<float2>* init_memory){

    writing_in_progress = false;
    dspace_rank = 2;
    dimsf = (hsize_t*)malloc(sizeof(hsize_t)*dspace_rank);

    //defining a complex data type for HDF5
    //NOTE: this is compatible with H5py/numpy interface
    complex_data_type = new CompType(sizeof(float2));
    complex_data_type->insertMember( "r", 0, PredType::NATIVE_FLOAT);
    complex_data_type->insertMember( "i", sizeof(float), PredType::NATIVE_FLOAT);


    stream_queue = init_queue;
    memory = init_memory;
}


//initialize after a streaming server class to terminate the dsp chain
H5_file_writer::H5_file_writer(Sync_server *streaming_server){

    writing_in_progress = false;
    dspace_rank = 2;
    dimsf = (hsize_t*)malloc(sizeof(hsize_t)*dspace_rank);

    //defining a complex data type for HDF5
    //NOTE: this is compatible with H5py/numpy interface
    complex_data_type = new CompType(sizeof(float2));
    complex_data_type->insertMember( "r", 0, PredType::NATIVE_FLOAT);
    complex_data_type->insertMember( "i", sizeof(float), PredType::NATIVE_FLOAT);


    stream_queue = streaming_server->out_queue;
    memory = streaming_server->memory;
}
void H5_file_writer::start(usrp_param* global_params){//

    if(not writing_in_progress){
        std::stringstream group_name;
        group_name<<"/raw_data"<<global_params->usrp_number;
        file = new H5File(get_name(),H5F_ACC_TRUNC);
        group = new Group( file->createGroup( group_name.str() ));

        //initialize the pointers to an invalid location by default (needed in conditions)
        A_TXRX=NULL;
        B_TXRX=NULL;
        A_RX2=NULL;
        B_RX2=NULL;

        int scalar_dspace_rank = 1;
        hsize_t* attribute_dimension = (hsize_t*)malloc(sizeof(hsize_t)*scalar_dspace_rank);
        attribute_dimension[0] = 1;
        DataSpace att_space_int(scalar_dspace_rank, attribute_dimension);
        DataSpace att_space_float(scalar_dspace_rank, attribute_dimension);

        Attribute err = group->createAttribute( "usrp_number", PredType::NATIVE_INT, att_space_int );
        err.write(PredType::NATIVE_INT, &global_params->usrp_number);

        int n_active_channels = global_params->get_number(RX);

        Attribute rx_num = group->createAttribute( "active_rx", PredType::NATIVE_INT, att_space_int );
        rx_num.write(PredType::NATIVE_INT, &n_active_channels);

        if(global_params->A_TXRX.mode != OFF){
            std::stringstream sub_group_name;
            sub_group_name<<group_name.str()<<"/A_TXRX";
            A_TXRX = new Group( file->createGroup( sub_group_name.str() ));
            write_properties(A_TXRX, &(global_params->A_TXRX));
        }
        if(global_params->B_TXRX.mode != OFF){
            std::stringstream sub_group_name;
            sub_group_name<<group_name.str()<<"/B_TXRX";
            B_TXRX = new Group( file->createGroup( sub_group_name.str() ));
            write_properties(B_TXRX, &(global_params->B_TXRX));
        }
        if(global_params->A_RX2.mode != OFF){
            std::stringstream sub_group_name;
            sub_group_name<<group_name.str()<<"/A_RX2";
            A_RX2 = new Group( file->createGroup( sub_group_name.str() ));
            write_properties(A_RX2, &(global_params->A_RX2));
        }
        if(global_params->B_RX2.mode != OFF){
            std::stringstream sub_group_name;
            sub_group_name<<group_name.str()<<"/B_RX2";
            B_RX2 = new Group( file->createGroup( sub_group_name.str() ));
            write_properties(B_RX2, &(global_params->B_RX2));
        }

        binary_writer = new boost::thread(boost::bind(&H5_file_writer::write_files,this));
    }else print_error("Cannot write a new binary file on disk, writing thread is still running.");
}

bool H5_file_writer::stop(bool force){
    //std::cout<<(force?"Closing HDF5 files..":"Waiting for HDF5 file(s)... ")<<std::flush;
    if(not force){
        force_close = false;
        binary_writer->interrupt();
        if (not writing_in_progress)binary_writer->join();
        return writing_in_progress;
    }else{
        force_close = true;
        binary_writer->interrupt();
        binary_writer->join();
        clean_queue(); //dangerous behaviour?
        return writing_in_progress;
    }
}

void H5_file_writer::close(){

    free(dimsf);

}

//in case of pointers update in the TXRX class method set() those functions have to be called
//this is because memory size adjustments on rx_output preallocator
void H5_file_writer::update_pointers(rx_queue* init_queue, preallocator<float2>* init_memory){
    stream_queue = init_queue;
    memory = init_memory;
}
void H5_file_writer::update_pointers(Sync_server *streaming_server){
    stream_queue = streaming_server->out_queue;
    memory = streaming_server->memory;
}

void H5_file_writer::write_properties(Group *measure_group, param* parameters_group){

    //set up attribute geometry
    int scalar_dspace_rank = 1;
    hsize_t* attribute_dimension = (hsize_t*)malloc(sizeof(hsize_t)*scalar_dspace_rank);
    attribute_dimension[0] = 1;
    DataSpace att_space_int(scalar_dspace_rank, attribute_dimension);

    //string type
    StrType str_type(PredType::C_S1, H5T_VARIABLE);

    std::vector<const char *> mode_c_str;
    mode_c_str.push_back(ant_mode_to_str(parameters_group->mode).c_str());
    Attribute mode(measure_group->createAttribute("mode" , str_type, att_space_int));
    mode.write(str_type, (void*)&mode_c_str[0]);

    //write scalar attributes
    Attribute rate = measure_group->createAttribute( "rate", PredType::NATIVE_INT, att_space_int );
    rate.write(PredType::NATIVE_INT, &(parameters_group->rate));

    Attribute rf = measure_group->createAttribute( "rf", PredType::NATIVE_INT, att_space_int );
    rf.write(PredType::NATIVE_INT, &(parameters_group->tone));

    Attribute gain = measure_group->createAttribute( "gain", PredType::NATIVE_INT, att_space_int );
    gain.write(PredType::NATIVE_INT, &(parameters_group->gain));

    Attribute bw = measure_group->createAttribute( "bw", PredType::NATIVE_INT, att_space_int );
    bw.write(PredType::NATIVE_INT, &(parameters_group->bw));

    Attribute samples = measure_group->createAttribute( "samples", PredType::NATIVE_LONG, att_space_int );
    samples.write(PredType::NATIVE_LONG, &(parameters_group->samples));

    Attribute delay = measure_group->createAttribute( "delay", PredType::NATIVE_FLOAT, att_space_int );
    delay.write(PredType::NATIVE_FLOAT, &(parameters_group->delay));

    Attribute burst_on = measure_group->createAttribute( "burst_on", PredType::NATIVE_FLOAT, att_space_int );
    burst_on.write(PredType::NATIVE_FLOAT, &(parameters_group->burst_on));

    Attribute burst_off = measure_group->createAttribute( "burst_off", PredType::NATIVE_FLOAT, att_space_int );
    burst_off.write(PredType::NATIVE_FLOAT, &(parameters_group->burst_off));

    Attribute buffer_len_att = measure_group->createAttribute( "buffer_len", PredType::NATIVE_INT, att_space_int );
    buffer_len_att.write(PredType::NATIVE_INT, &(parameters_group->buffer_len));

    Attribute fft_tones = measure_group->createAttribute( "fft_tones", PredType::NATIVE_INT, att_space_int );
    fft_tones.write(PredType::NATIVE_INT, &(parameters_group->fft_tones));

    Attribute pf_average = measure_group->createAttribute( "pf_average", PredType::NATIVE_INT, att_space_int );
    pf_average.write(PredType::NATIVE_INT, &(parameters_group->pf_average));

    Attribute decim = measure_group->createAttribute( "decim", PredType::NATIVE_INT, att_space_int );
    decim.write(PredType::NATIVE_INT, &(parameters_group->decim));

    //set the number of channels
    int n_chan_int = (int)(parameters_group->wave_type.size());//address of temporary is a bad isea
    Attribute n_chan = measure_group->createAttribute( "n_chan", PredType::NATIVE_INT, att_space_int );
    n_chan.write(PredType::NATIVE_INT, &n_chan_int);

    //vectorial dataspaces geometry setup
    hsize_t* attribute_dimension_vect = (hsize_t*)malloc(sizeof(hsize_t)*scalar_dspace_rank);
    attribute_dimension_vect[0] = parameters_group->wave_type.size();
    DataSpace vect_dataspace(scalar_dspace_rank, attribute_dimension_vect);

    //Convert the vector into a C string array.
    //Because the input function ::write requires that.
    std::vector<const char *> cStrArray;
    for(size_t index = 0; index < parameters_group->wave_type.size(); ++index){
        cStrArray.push_back(w_type_to_str(parameters_group->wave_type[index]).c_str());
    }

    //write the description of the signal processing
    Attribute wave_types(measure_group->createAttribute("wave_type" , str_type, vect_dataspace));
    wave_types.write(str_type, (void*)&cStrArray[0]);

    Attribute freq = measure_group->createAttribute( "freq", PredType::NATIVE_INT, vect_dataspace );
    freq.write(PredType::NATIVE_INT, (void*)&(parameters_group->freq)[0]);

    Attribute chirp_f = measure_group->createAttribute( "chirp_f", PredType::NATIVE_INT, vect_dataspace );
    chirp_f.write(PredType::NATIVE_INT, (void*)&(parameters_group->chirp_f)[0]);

    Attribute swipe_s = measure_group->createAttribute( "swipe_s", PredType::NATIVE_INT, vect_dataspace );
    swipe_s.write(PredType::NATIVE_INT, (void*)&(parameters_group->swipe_s)[0]);

    Attribute ampl = measure_group->createAttribute( "ampl", PredType::NATIVE_FLOAT, vect_dataspace );
    ampl.write(PredType::NATIVE_FLOAT, (void*)&(parameters_group->ampl)[0]);

    Attribute chirp_t = measure_group->createAttribute( "chirp_t", PredType::NATIVE_FLOAT, vect_dataspace );
    chirp_t.write(PredType::NATIVE_FLOAT, (void*)&(parameters_group->chirp_t)[0]);

}

std::string H5_file_writer::get_name(){

    time_t rawtime;
    struct tm * timeinfo;
    char buffer[100];
    std::stringstream ss;

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer,sizeof(buffer),"%d%m%Y_%I%M%S",timeinfo);
    std::string str(buffer);

    ss << "USRP_"<<str<<".h5";
    std::cout<<"Saving measure to file name: "<<ss.str()<<std::endl;
    return ss.str();

}
void H5_file_writer::clean_queue(){

    RX_wrapper wrapper;
    if(not writing_in_progress){
        while(not stream_queue->empty())if(stream_queue->pop(wrapper))memory->trash(wrapper.buffer);
    }else print_warning("Cannot clean the binary file queue: binary file writer is still runnning.");
}

void H5_file_writer::write_files(){

    bool active = true;
    writing_in_progress = true;
    RX_wrapper wrapper;

    //hdf5 attributes for single buffer
    int scalar_dspace_rank = 1;
    hsize_t* attribute_dimension = (hsize_t*)malloc(sizeof(hsize_t)*scalar_dspace_rank);
    attribute_dimension[0] = 1;
    DataSpace att_space(scalar_dspace_rank, attribute_dimension);

    //variable needed to
    bool finishing = true;
    while(active or finishing){
        try{
            boost::this_thread::interruption_point();

            if(stream_queue->pop(wrapper)){
                //select subgroup where to write the packet
                std::stringstream ss;
                ss<<"raw_data"<<wrapper.usrp_number<<"/"<< get_front_end_name(wrapper.front_end_code) <<"/dataset_"<<wrapper.packet_number<<"/";

                // create new dspace (holds geometry info about the data)
                //NOTR: rank is always 2.
                dimsf[0] = wrapper.channels;
                dimsf[1] = wrapper.length/wrapper.channels;
                dataspace = new DataSpace(dspace_rank, dimsf);

                //data container in the group
                DataSet dataset = file->createDataSet( ss.str(), *complex_data_type, *dataspace );


                //write attributes
                Attribute pn = dataset.createAttribute( "packet_number", PredType::NATIVE_INT, att_space );
                pn.write(PredType::NATIVE_INT, &wrapper.packet_number);

                Attribute err = dataset.createAttribute( "errors", PredType::NATIVE_INT, att_space );
                err.write(PredType::NATIVE_INT, &wrapper.errors);

                //write data in the dataset
                dataset.write( wrapper.buffer, *complex_data_type );
                dataset.flush(H5F_SCOPE_LOCAL);

                //cleanup stuff
                delete dataspace;

                memory->trash(wrapper.buffer);
            }else{
                boost::this_thread::sleep_for(boost::chrono::milliseconds{20});
            }

            if(not active)finishing = not stream_queue->empty();

        }catch(boost::thread_interrupted &){
            active = false;
            if (not force_close){
                finishing = not stream_queue->empty();
            }else finishing = false;
        }
    }
    if(A_TXRX){
        A_TXRX->flush(H5F_SCOPE_LOCAL);
        A_TXRX->flush(H5F_SCOPE_GLOBAL);
        delete A_TXRX;
    }
    if(A_RX2){
        A_RX2->flush(H5F_SCOPE_LOCAL);
        A_RX2->flush(H5F_SCOPE_GLOBAL);
        delete A_RX2;
    }
    if(B_TXRX){
        B_TXRX->flush(H5F_SCOPE_LOCAL);
        B_TXRX->flush(H5F_SCOPE_GLOBAL);
        delete B_TXRX;
    }
    if(B_RX2){
        B_RX2->flush(H5F_SCOPE_LOCAL);
        B_RX2->flush(H5F_SCOPE_GLOBAL);
        delete B_RX2;
    }


    group->flush(H5F_SCOPE_LOCAL);
    group->flush(H5F_SCOPE_GLOBAL);
    file->flush(H5F_SCOPE_LOCAL);
    file->flush(H5F_SCOPE_GLOBAL);

    delete group;
    delete file;

    writing_in_progress = false;
}

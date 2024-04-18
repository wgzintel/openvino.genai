# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os


def save_text_to_file(input_text, text_file_name, args):
    if args['output_dir'] is not None:
        if os.path.exists(args['output_dir']) is False:
            os.mkdir(args['output_dir'])
        out_path = args['output_dir']
    else:
        out_path = '.'
    save_path = out_path + os.sep + text_file_name
    input_text_file = open(save_path, 'w')
    input_text_file.write(input_text)
    input_text_file.close()


def save_image_file(img, img_file_name, args):
    if args['output_dir'] is not None:
        if os.path.exists(args['output_dir']) is False:
            os.mkdir(args['output_dir'])
        out_path = args['output_dir']
    else:
        out_path = '.'
    save_path = out_path + os.sep + img_file_name
    img.save(save_path)
    return save_path


def output_input_text(input_text, args, model_precision, prompt_idx, batchsize_idx, start_of_log):
    out_str = replace_brackets_with_underscores(start_of_log)
    if args['batch_size'] > 1:
        text_file_name = args['model_name'] + '_' + model_precision + '_p' + str(prompt_idx) + '_bs' + str(batchsize_idx)
    else:
        text_file_name = args['model_name'] + '_' + model_precision + '_p' + str(prompt_idx)
    text_file_name = text_file_name + out_str + 'input.txt'
    save_text_to_file(input_text, text_file_name, args)


def output_image_input_text(input_text, args, prompt_idx, batchsize_idx, start_of_log):
    out_str = replace_brackets_with_underscores(start_of_log)
    if args['batch_size'] > 1 and batchsize_idx is not None:
        text_file_name = args['model_name'] + '_p' + str(prompt_idx) + '_bs' + str(batchsize_idx)
    else:
        text_file_name = args['model_name'] + '_p' + str(prompt_idx)
    text_file_name = text_file_name + out_str + 'input.txt'
    save_text_to_file(input_text, text_file_name, args)


def output_gen_text(generated_text, args, model_precision, prompt_idx, iteration, batchsize_idx, start_of_log):
    out_str = replace_brackets_with_underscores(start_of_log)
    if args['batch_size'] > 1:
        text_file_name = args['model_name'] + '_' + model_precision + '_p' + str(prompt_idx) + '_bs' + str(batchsize_idx)
    else:
        text_file_name = args['model_name'] + '_' + model_precision + '_p' + str(prompt_idx)
    text_file_name = text_file_name + '_iter' + str(iteration) + out_str + 'output.txt'
    save_text_to_file(generated_text, text_file_name, args)


def output_gen_image(img, args, prompt_idx, iteration, batchsize_idx, suffix, start_of_log):
    out_str = replace_brackets_with_underscores(start_of_log)
    if args['batch_size'] > 1 and batchsize_idx is not None:
        img_save_name = args['model_name'] + '_p' + str(prompt_idx) + '_bs' + str(batchsize_idx)
    else:
        img_save_name = args['model_name'] + '_p' + str(prompt_idx)
    img_save_name = img_save_name + '_iter' + str(iteration) + out_str + 'output' + suffix
    img_save_path = save_image_file(img, img_save_name, args)
    return img_save_path


def replace_brackets_with_underscores(input_str):
    out_str = input_str.replace('][', '_')
    out_str = out_str.replace('[', '_')
    out_str = out_str.replace(']', '_')
    return out_str
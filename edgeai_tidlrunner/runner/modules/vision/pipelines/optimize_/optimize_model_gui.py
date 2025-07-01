# Copyright (c) 2018-2025, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import os
import shutil

from ...settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from ..... import utils
from ..... import bases


class OptimizeModelGUIPipeline(bases.PipelineBase):
    ARGS_DICT=SETTINGS_DEFAULT['basic']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['basic']
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
            
    def run(self):
        print(f'INFO: starting model optimize_gui')
        try:
            import streamlit as st
            import pandas as pd
            os.system(f"streamlit run {__file__} -- --model_path={self.kwargs['model_path']} --run_dir={self.kwargs['run_dir']}")
        except:
            raise RuntimeError("install requirements via: pip3 install streamlit pandas")


    def on_click_run_optimize(self, model_path, output_path, **kwargs):
        if os.path.exists(output_path):
            print(f'INFO: clearing run_dir folder before compile: {output_path}')
            shutil.rmtree(output_path, ignore_errors=True)
        #
        from osrt_model_tools.onnx_tools.tidl_onnx_model_optimizer.optimize import get_optimizers, optimize
        optimize(model_path, output_path, **kwargs)


    def main(self, **kwargs):
        run_dir = kwargs.pop('run_dir')
        run_dir = run_dir.replace('{model_name}', os.path.splitext(os.path.basename(kwargs['model_path']))[0])
        model_folder = os.path.join(run_dir, 'model')

        # if os.path.exists(run_dir):
        #     print(f'INFO: clearing run_dir folder before compile: {run_dir}')
        #     shutil.rmtree(run_dir, ignore_errors=True)
        # #

        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(model_folder, exist_ok=True)

        model_path = os.path.join(model_folder, os.path.basename(kwargs['model_path']))

        input_path = os.path.abspath(kwargs['model_path'])
        output_path = os.path.abspath(model_path)

        ##############################################
        import streamlit as st
        import pandas as pd

        st.markdown(
            """
            <style>
            .main .css-1a2j2n6 {
                text-align: left;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        if 'model_path' not in st.session_state:
            st.session_state.model_path = input_path #'.'

        if 'is_clicked' not in st.session_state:
            st.session_state.is_clicked = False

        st.sidebar.title("Optimizer GUI")

        st.sidebar.write("Hi, this is the Optimizer GUI.")

        folder_path = '.' if st.session_state.model_path == '' else st.session_state.model_path
        if not os.path.exists(folder_path):
            folder_path = '.'

        folder_path = os.path.abspath(folder_path)

        if os.path.isdir(folder_path):
            file_names = [p for p in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, p)) or p.endswith('.onnx')] + ['..']

            selected_path = st.sidebar.selectbox(
                'Choose your model', file_names, index=0, format_func=lambda x: x if x != '..' else 'Parent Directory'
            )
            if selected_path == '..':
                st.session_state.model_path, _ = os.path.split(folder_path)
            else:
                st.session_state.model_path = os.path.join(folder_path, selected_path)
        else:
            folder_path, file_name = os.path.split(folder_path)
            file_names = [p for p in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, p)) or p.endswith('.onnx')] + ['..']
            selected_path = st.sidebar.selectbox(
                'Choose your model', file_names, index= file_names.index(file_name), format_func=lambda x: x if x != '..' else 'Parent Directory'
            )
            if selected_path == '..':
                st.session_state.model_path, _ = os.path.split(folder_path)
            else:
                st.session_state.model_path = os.path.join(folder_path, selected_path)

        from osrt_model_tools.onnx_tools.tidl_onnx_model_optimizer.optimize import get_optimizers, optimize
        try:
            from  osrt_model_tools.onnx_tools.tidl_onnx_model_optimizer.optimize import print_node_count_table
        except:
            print("WARNING: print_node_count_table is not found in osrt_model_tools")
            print_node_count_table = None

        import onnx
        model_path = st.text_input("Model Path", st.session_state.model_path)

        def add_options():
            optimizer = {}
            for op, val in get_optimizers().items():
                if op == 'simplify_kwargs':
                    optimizer[op] = val
                elif op in ('shape_inference_mode', 'simplify_mode'):
                    options = ['pre', 'post', 'all', None]
                    mode = st.selectbox(op, [opt if opt else 'None' for opt in options], index=options.index(val))
                    optimizer[op] = mode
                else:
                    optimizer[op] = st.checkbox(op, key=op, value=val)
            return optimizer

        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"

        def filter_data(data):
            for row in data:
                for i, elm in enumerate(row):
                    if isinstance(elm, str):
                        row[i] = elm.replace(RED, "").replace(GREEN, "").replace(RESET, "")
            return data

        if os.path.exists(model_path) :
            if os.path.isfile(model_path):
                if model_path.endswith('.onnx'):
                    folder_path, file_name = os.path.split(model_path)
                    optimized_model_name = st.text_input("Optimized Model Name", output_path)
                    # optimized_model_name = os.path.join(folder_path, optimized_model_name)
                    print(f'INFO: optimized_model_name={optimized_model_name}')
                    optimizer = add_options()
                    if (not st.session_state.is_clicked) and os.path.exists(optimized_model_name):
                        st.warning(f"File already exists: {optimized_model_name}")
                        print((model_path, optimized_model_name))
                    st.session_state.is_clicked = st.button("Optimize", on_click=self.on_click_run_optimize, args=(model_path, optimized_model_name), kwargs=dict(custom_optimizer=optimizer))
                    if st.session_state.is_clicked and os.path.exists(optimized_model_name):
                        st.success(f"Optimized model saved to {optimized_model_name}")
                        if print_node_count_table:
                            st.write("Node count comparison:")
                            headers = ["Operation", "Original Model", "Optimized Model"]
                            data = print_node_count_table(onnx.load(model_path), onnx.load(optimized_model_name))
                            data = filter_data(data)
                            data = pd.DataFrame( data,columns=headers,)
                            st.dataframe(data, hide_index=True)
                        #
                else:
                    st.warning(f"{model_path} is not an onnx model" )
                    folder_path, file_name = os.path.split(model_path)
                    st.session_state.model_path = folder_path
            else:
                st.session_state.model_path = model_path
        else:
            st.warning(f"{model_path} does not exist" )
            st.session_state.model_path = '.'


if __name__ == '__main__':
    optimizer = OptimizeGUIPipeline()
    parser = optimizer.get_arg_parser()
    args = parser.parse_args()
    kwargs = vars(args)
    optimizer.main(**kwargs)

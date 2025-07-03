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
import sys
import shutil
import onnx
import onnx_graphsurgeon as gs
import os
from ..... import utils
from ..... import bases
from ...settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT


#################################################################
import onnx
import onnx_graphsurgeon as gs
import os


class ONNXNode:
    def __init__(self):
        self.name = None
        self.module = None
        self.children = None
        self.depth = None
        self.pattern_id = None
        self.no_of_nodes = None

    def __repr__(self):
        return f"{self.name} : {self.module} : {self.depth} : {[child.name if isinstance(child,ONNXNode) else child for child in self.children]}"
        

class SplitModel(bases.PipelineBase):
    ARGS_DICT=SETTINGS_DEFAULT['basic']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['basic']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def info(self):
        print(f'INFO: Model optimize - {__file__}')

    def _get_nodes_start_with(self, graph, prefix):
        return [node for node in graph.nodes if node.name.startswith(prefix)]

    def _export_submodule_start_with(self, onnx_path:str,prefix:str,export_path:str=None):
        '''
        onnx export the submodule of the onnx model to a new onnx file
        ( here submodule means the nodes whose name starts with the prefix)
        args:
            onnx_path:      the path of the onnx model
            prefix:         the prefix of the submodule nodes
                            prefix must start and end with '/'
                            as '/' serves as a separator between a module and its submodules a node's name
            export_path:    the path of the new onnx model
                            default :None (export filename: original file name + prefix name)
        '''
        if export_path is None:
            directory, file = os.path.split(onnx_path)
            file_name, ext =os.path.splitext(file)
            directory = os.path.join(directory,file_name)
            file_name = f"{prefix.replace('/','_').replace('.','__')}"
            file = file_name + ext
            export_path = os.path.join(directory,file)

        model = onnx.load(onnx_path)
        graph = gs.import_onnx(model)
        original_inputs = list(graph.inputs)
        original_outputs = list(graph.outputs)
        nodes = self._get_nodes_start_with(graph, prefix)
        graph.inputs = []
        graph.outputs = []
        for node in nodes:
            for inp in node.inputs:
                if isinstance(inp, gs.Constant):
                    continue
                if len(inp.inputs) >0 and inp.inputs[0] not in nodes:
                    graph.inputs.append(inp) if inp not in graph.inputs else None
                elif (inp in original_inputs):
                    graph.inputs.append(inp) if inp not in graph.inputs else None
            for out in node.outputs:
                if any(output not in nodes for output in out.outputs) :
                    graph.outputs.append(out) if out not in graph.outputs else None
                elif (out in original_outputs):
                    graph.outputs.append(out) if out not in graph.outputs else None
        rest_nodes = [node for node in graph.nodes if node not in nodes]
        for node in rest_nodes:
            graph.nodes.remove(node)
        graph.cleanup()
        graph.toposort()
        final_model = gs.export_onnx(graph)
        directory, file = os.path.split(export_path)
        if not os.path.exists(directory) and directory != '':
            os.makedirs(directory)
        onnx.save_model(final_model, export_path)
        return export_path


    def _export_all_top_level_submodules(self, onnx_path:str):
        '''
        onnx exports all top level submodules of the onnx model to new onnx files
        it discards stand alone nodes  which are not part of any submodule
        args:
            onnx_path:      the path of the onnx model
        '''
        model = onnx.load(onnx_path)
        graph = gs.import_onnx(model)
        top_level_submodule_names = {node.name.split('/')[1] for node in graph.nodes if '/' in node.name[1:]}
        export_paths = [self._export_submodule_start_with(onnx_path,f'/{name}/') for name in top_level_submodule_names]
        return export_paths


    #################################################################
    def _get_submodule_dict(self, name:str):
        names = name.split('/')[:-1]
        submodule_dict = {}
        for i, name in enumerate(names[:-1]):
            submodule_dict['/'.join(names[:i+1])+'/'] = (names[i+1],'/'.join(names[:i+2])+ '/')
        return submodule_dict


    def _export_unique_submodules(self, onnx_path:str, output_path:str, max_depth = 3):
        model = onnx.load(onnx_path)
        graph = gs.import_onnx(model)
        directory = output_path
        _,file = os.path.split(onnx_path)
        file_name, ext = os.path.splitext(file)
        output_file = os.path.join(output_path, 'submodule_tree.txt')
        node_names = [node.name for node in graph.nodes]
        # get all the unique submodules
        submodule_names = []
        for node in graph.nodes:
            modules = node.name.split('/')[:-1]
            for i, module in enumerate(modules):
                if module == '':
                    continue
                if i == 0:
                    submodule_names.append('/'+module+'/')
                else:
                    submodule_names.append('/'.join(modules[:i+1])+'/')
        submodule_names = list(set(submodule_names))
        submodule_nodes = {name:self._get_nodes_start_with(graph,name) for name in submodule_names}
        to_be_removed = [name for name, nodes in submodule_nodes.items() if len(nodes)==1]
        for name in to_be_removed:
                submodule_nodes.pop(name)

        submodule_op_lists = {name:','.join([node.op for node in nodes])for name,nodes in submodule_nodes.items()}

        unique_op_lists = {}
        for name,op_list in submodule_op_lists.items():
            if op_list in unique_op_lists:
                unique_op_lists[op_list].append(name)
            else:
                unique_op_lists[op_list] = [name]
        submodule_op_dict = {}
        for i, (op_list, names) in enumerate(unique_op_lists.items()):
            for name in names:
                submodule_op_dict[name] = i+1
        onnx_paths = [onnx_path]
        for i, op_list in enumerate(unique_op_lists):
            onnx_paths.append(self._export_submodule_start_with(onnx_path, unique_op_lists[op_list][0],os.path.join(directory,f'pattern_{i+1}.onnx')))

        submodule_dict = {}
        for name in submodule_names:
            if name in to_be_removed:
                continue
            temp = self.get_submodule_dict(name)
            for name,submodule in temp.items():
                if name in submodule_dict:
                    submodule_dict[name].append(submodule) if submodule not in submodule_dict[name] else None
                else:
                    submodule_dict[name] = [submodule]


        queue = ['/']
        nodes = {}
        while queue:
            module = queue.pop(0)
            node = ONNXNode()
            node.module = module
            node.no_of_nodes = len(submodule_nodes.get(module,[]))
            node.children = [submodule for submodule in submodule_dict.get(module,[])]
            nodes[module]= node
            node.pattern_id = submodule_op_dict.get(module,0)
            queue.extend([submodule[1] for submodule in submodule_dict.get(module,[])])

        nodes['/'].name = 'root'
        nodes['/'].depth = 0
        nodes['/'].no_of_nodes = len(graph.nodes)
        for module,node in nodes.items():
            child_nodes = []
            if isinstance(node.children,list):
                for child in node.children:
                    name,child_module = child
                    child_node = nodes[child_module]
                    child_node.name = name
                    child_node.depth = node.depth + 1
                    child_nodes.append(child_node)
            node.children = sorted(child_nodes, key= lambda n: node_names.index(submodule_nodes[n.module][0].name))

        with open(output_file,'w') as f:
            stack = [nodes['/']]
            while stack:
                node = stack.pop()
                if node.depth > max_depth:
                    continue
                s = ''
                for _ in range(node.depth):
                    s += '    '
                s += node.name + ' : ' + node.module +f' (pattern: {node.pattern_id})'+ f' (No. of Nodes: {node.no_of_nodes})'+ '\n'
                f.write(s)
                stack.extend(reversed(node.children))


    #################################################################
    def _run(self):
        print(f'INFO: starting split_model')
        run_dir = self.kwargs.pop('run_dir')
        run_dir = run_dir.replace('{model_name}', os.path.splitext(os.path.basename(self.kwargs['model_path']))[0])
        model_folder = os.path.join(run_dir, 'model')

        if os.path.exists(run_dir):
            print(f'INFO: clearing run_dir folder before compile: {run_dir}')
            shutil.rmtree(run_dir, ignore_errors=True)
        #

        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(model_folder, exist_ok=True)

        output_path = model_folder

        print(f'INFO: running split_model')
        self._export_unique_submodules(self.kwargs['model_path'], output_path, max_depth=3)


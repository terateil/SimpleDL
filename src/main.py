import pickle

import tkinter as tk
from tkinter import *
import tkinter.ttk as ttk
import tkinter.messagebox as msgbox
import tkinter.filedialog

import os
import sys
import datetime

import keras
import tensorflow.keras
import numpy as np

class SimpleDL(tkinter.Tk): #main app.
    def __init__(self):
        super().__init__()
        self.title("simpleDL")

        self.width = 640
        self.height = 480
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        self.geometry(f"{self.width}x{self.height}+{int(self.screen_width / 2 - self.width / 2)}+{int(self.screen_height / 2 - self.height / 2)}")
        self.resizable(0, 0)

        self.title_label = Label(self, text='환영합니다', font=("Arial", 40))
        self.title_label.pack(side='top', expand=True)
        self.load_model_btn = Button(self, text="모델 불러오기", command=self.launch_modelUser, font=("Arial", 30))
        self.load_model_btn.pack(side='top', expand=True)
        self.make_new_model_btn = Button(self, text="새 모델 생성", command=self.launch_ModelBaseMaker, font=("Arial", 30))
        self.make_new_model_btn.pack(side='top', expand=True)

    def launch_ModelBaseMaker(self):
        modelMaker = DLModelBaseMaker(self)
        self.attributes("-disabled", True)
        modelMaker.lift()
        modelMaker.focus_force()
        modelMaker.grab_set()
        #modelMaker.attributes("-topmost", True)

    def launch_modelUser(self):
        model_folder = tkinter.filedialog.askopenfilename(initialdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'),  filetypes = (("model file", "*.h5"),),  defaultextension='.h5', parent = self)
        if len(model_folder) > 0:
            modelUser = ModelUser(self, model_folder)
            self.attributes("-disabled", True)
            modelUser.lift()
            modelUser.focus_force()
            modelUser.grab_set()


class DLModelBaseMaker(tkinter.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent

        self.title("모델 생성")

        self.width = 600
        self.height = 400
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        self.geometry(f"{self.width}x{self.height}+{int(self.screen_width / 2 - self.width / 2)}+{int(self.screen_height / 2 - self.height / 2)}")

        self.protocol("WM_DELETE_WINDOW", self.closing)
        #####################################################################################

        self.model_type_info_frame = LabelFrame(self, text="1. 모델 종류를 선택하세요.")
        self.model_type_info_frame.pack(fill="x", padx=5, pady=5, ipady=10)
        model_types = ["회귀", "분류", "직접 설계"]
        output_text_dict = {'회귀': '1', '분류': '분류할 클래스 개수 입력', "직접 설계":''} #직접설계, 필요 없음.
        self.model_type_comboBox = ttk.Combobox(self.model_type_info_frame, state="readonly", values=model_types)
        self.model_type_comboBox.pack(side="left")
        self.model_type_comboBox.bind("<<ComboboxSelected>>", lambda event: self.set_output_size_text(self.output_size_entry,
            output_text_dict[self.model_type_comboBox.get()]))

        #############################################################################################

        self.model_input_info_frame = LabelFrame(self, text="2. 모델의 입력 데이터 정보를 입력하세요.")
        self.model_input_info_frame.pack(fill="x", padx=5, pady=5, ipady=10)
        data_types = ["일반", "이미지", "시계열"] #->dynamic
        self.input_type_comboBox = ttk.Combobox(self.model_input_info_frame, state="readonly", values=data_types)
        self.input_type_comboBox.pack(side="left")
        self.input_type_comboBox.bind("<<ComboboxSelected>>", lambda event: self.pack_and_unpack(self.input_type_comboBox.get()) )

        ###############################################################################################

        self.model_output_info_frame = LabelFrame(self, text="3. 모델의 출력 데이터 정보를 입력하세요.")
        self.model_output_info_frame.pack(fill="x", padx=5, pady=5, ipady=10)
        self.output_size_entry = Entry(self.model_output_info_frame)
        self.output_size_entry.pack(side="left")
        self.helpText_dict = {'회귀': None, '분류': '분류할 클래스 개수 입력', "직접 설계":None}
        self.output_size_entry.bind("<FocusIn>", lambda event: self.control_entry_helptext(self.output_size_entry, True,
                                                                                            None))
        self.output_size_entry.bind("<FocusOut>", lambda event: self.control_entry_helptext(self.output_size_entry, False,
                                                                                            self.helpText_dict[self.model_type_comboBox.get()]))

        self.btn_frame = Frame(self)
        self.btn_frame.pack(fill="x", padx=5, pady=5)

        self.cancel_btn = Button(self.btn_frame, text="취소", width=10, command=self.closing)
        self.cancel_btn.pack(side="right", padx=5, pady=5)

        self.confirm_btn = Button(self.btn_frame, text="생성", width = 10, command=self.show_confirmcheck_msg)
        self.confirm_btn.pack(side="right",padx=5, pady=5)


    def pack_and_unpack(self, data_type):
        for info_cmb in self.model_input_info_frame.winfo_children()[1:]:
            info_cmb.destroy()

        if data_type == "일반":
            feature_cnt_label = Label(self.model_input_info_frame, text="특성 개수")
            feature_cnt_label.pack(side="left")
            feature_cnt_entry = Entry(self.model_input_info_frame)
            feature_cnt_entry.pack(side="left")

        elif data_type == "이미지":
            width_label = Label(self.model_input_info_frame, text="가로(px)")
            width_label.pack(side = "left")
            width_entry = Entry(self.model_input_info_frame, width = 5)
            width_entry.pack(side = "left")
            height_label = Label(self.model_input_info_frame, text="세로(px)")
            height_label.pack(side = "left")
            height_entry = Entry(self.model_input_info_frame, width = 5)
            height_entry.pack(side = "left")
            channel_label = Label(self.model_input_info_frame, text="채널")
            channel_label.pack(side = "left")
            channel_cmb = ttk.Combobox(self.model_input_info_frame, state="readonly", values=["컬러(3)", "흑백(1)"], width = 10)
            channel_cmb.pack(side = "left")

        else: #시계열
            feature_cnt_label = Label(self.model_input_info_frame, text="특성 개수")
            feature_cnt_label.pack(side="left")
            feature_cnt_entry = Entry(self.model_input_info_frame)
            feature_cnt_entry.pack(side="left")


    def fix_combobox(self, cmb: ttk.Combobox, value):
        cmb.set(value)
        cmb.state("disabled")

    def set_output_size_text(self, entry, value):
        entry.config(state="normal")
        entry.delete(0, END)
        entry.insert(0, value)
        if value == '1' or value == '':
            entry.config(state="disabled")
        else:
            entry.config(fg='grey')

    def control_entry_helptext(self, entry:Entry, focus ,text):
        entry.config(state="normal")
        if focus and entry.get() == self.helpText_dict['분류']:
            entry.delete(0, END)
            entry.config(fg='black')
        elif not focus and entry.get() == '':
            entry.insert(0, text)
            entry.config(fg='grey')


    def closing(self):
        self.parent.attributes("-disabled", False)
        self.destroy()

    def show_confirmcheck_msg(self):
        confirm = msgbox.askokcancel(title="확인", message="입력한 조건으로 모델 베이스를 생성합니다. 생성하시겠습니까?", parent=self)
        if confirm:
            self.launch_DLModelDesigner()

    def get_model_base_data(self):
        data={}
        data['model_type'] = self.model_type_comboBox.get()
        data['input_type'] = self.input_type_comboBox.get()
        data['input'] = [1 if d.get()=='흑백(1)' else (3 if d.get()=='컬러(3)' else int(d.get()))
                         for d in self.model_input_info_frame.winfo_children()[2::2]]
        data['output'] = int(self.output_size_entry.get()) if self.output_size_entry.get()!='' else None
        return data

    def launch_DLModelDesigner(self):
        modelDesigner = DLModelDesigner(self, self.get_model_base_data())
        self.attributes("-disabled", True)
        modelDesigner.lift()
        modelDesigner.focus_force()
        modelDesigner.grab_set()
        # modelMaker.attributes("-topmost", True)



class DLModelDesigner(tkinter.Toplevel): #main. model designing window.
    def __init__(self, parent, model_base_data):
        super().__init__(parent)

        self.width = 1500
        self.height = 800
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        self.geometry(
            f"{self.width}x{self.height}+{int(self.screen_width / 2 - self.width / 2)}+{int(self.screen_height / 2 - self.height / 2)}")

        self.protocol("WM_DELETE_WINDOW", self.closing)

        #print(model_base_data)

        self.parent = parent
        self.model_base_data = model_base_data


        self.model_type = model_base_data['model_type']
        self.input_type = model_base_data['input_type']
        if self.input_type == '시계열':
            self.input_shape = [None] + model_base_data['input']
        else:
            self.input_shape = model_base_data['input']
        self.output_shape = model_base_data['output']

        self.input = tensorflow.keras.Input(self.input_shape)
        self.layers = []


        ################################################

        self.scrollableframe = ScrollableFrame(self)
        self.scrollableframe.pack(side='left', padx = 10)
        self.layerListFrame = self.scrollableframe.win

        self.layers.append(LayerWidget(self.layerListFrame, type='Input'))
        if self.output_shape is not None:
            self.layers.append(LayerWidget(self.layerListFrame, type='Dense', fixed_out=True))
            self.layers.append(LayerWidget(self.layerListFrame, type='Output'))
        self.update_layerList()


        #########################3
        self.modelnameFrame = Frame(self)
        self.modelnameFrame.pack(side='right', fill='x')
        self.model_name_Label = Label(self.modelnameFrame, text='모델 이름(영문): ')
        self.model_name_Label.pack(side="left", padx=5)
        self.model_name_entry = Entry(self.modelnameFrame)
        self.model_name_entry.pack(side="left", padx=5)

        self.model_build_btn = Button(self.modelnameFrame, text='생성', command=self.build_model)
        self.model_build_btn.pack(side='bottom',pady=5)



    def build_model(self):
        data = tensorflow.identity(self.input)
        for l in self.layers:
            if l.type in ['Input', 'Output']:
                continue

            info = [d.get() for d in l.infoFrame.winfo_children()[1:-1:2]]
            info.append(l.infoFrame.winfo_children()[-1].get() if l.type == 'Dense' else l.boolVar.get())
            #print(info)

            layer = None
            if l.type == 'Dense':
                layer = tensorflow.keras.layers.Dense(int(info[0]), activation = None if info[1]=='없음' else info[1])
            elif l.type == 'Conv2D':
                layer = tensorflow.keras.layers.Conv2D(int(info[0]), (int(info[1]), int(info[2])), #strides = (int(info[3]), int(info[4])),
                                                        )

                if info[3] =='Max Pooling':
                    data=layer(data)
                    layer = tensorflow.keras.layers.MaxPooling2D()
                elif info[3] == 'Mean Pooling':
                    data=layer(data)
                    layer = tensorflow.keras.layers.MeanPooling2D()

                if info[4]:
                    data=layer(data)
                    layer = tensorflow.keras.layers.Flatten()
            elif l.type == 'RNN':
                layer = tensorflow.keras.layers.SimpleRNN(int(info[0]), return_sequences = info[1])

            data = layer(data)


        model = tensorflow.keras.Model(self.input, data)
        model.compile()

        confirm = msgbox.askyesno(title="모델 생성 완료", message="모델을 생성 완료했습니다. 저장하시겠습니까?", parent=self)
        save_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        if confirm:
            model_name=self.model_name_entry.get()
            if len(model_name) > 0:
                model_dir = os.path.join(save_base_dir, model_name)
                if os.path.isdir(model_dir):
                    msgbox.showwarning(title = '모델명 중복', message='이미 존재하는 모델 이름입니다. 다른 이름을 입력하세요.', parent=self)
                else:
                    os.mkdir(model_dir)
                    tensorflow.keras.models.save_model(model, os.path.join(model_dir, f'{model_name}.h5'))
                    with open(os.path.join(model_dir, f'{model_name}.pickle'), 'wb') as data_path:
                        pickle.dump(self.model_base_data, data_path)

                    msgbox.showinfo(title='저장 완료', message="모델 저장이 완료되었습니다.", parent=self)




    def deleteLayer(self, layer):
        for l in self.layers:
            if l is layer:
                self.layers.remove(l)
                self.update_layerList()
                break

    def addLayer(self, layer, type):
        for index, l in enumerate(self.layers):
            if l is layer:
                self.layers.insert(index+1, LayerWidget(self.layerListFrame, type))
                self.update_layerList()
                break

    def update_layerList(self):
        for layerWidget in self.layerListFrame.winfo_children():
            layerWidget.grid_remove()
        for index, layer in enumerate(self.layers):
            layer.grid(row=index, column=0, pady=10, )


    def closing(self):
        self.parent.attributes("-disabled", False)
        self.destroy()


class LayerWidget(tk.LabelFrame):
    def __init__(self, parent, type, **kwargs):
        #print(kwargs)
        self.parent = parent
        self.scrollableframe = parent.master.master
        self.designer = self.scrollableframe.master
        super().__init__(parent, height=100)


        self.type = type
        self.deleteable = True
        self.addable = True


        self.typeLabel = Label(self, text = self.type + (' Layer' if self.type not in ['Input', 'Output'] else '') )
        self.typeLabel.config(anchor=CENTER)
        self.typeLabel.grid(row=0, column=0, sticky='WE')

        self.infoFrame = Frame(self)
        self.infoFrame.grid(row=1, column=0, sticky='WE')

        self.pad_x=5

        if self.type == 'Input':
            self.deleteable = False
            self.typeLabel.config(fg='red')
            self.input_size_label = Label(self.infoFrame, text = f'입력 크기 : {self.designer.input_shape}')
            self.input_size_label.pack(side="left", padx=self.pad_x)

        if self.type == 'Dense':
            self.unit_label = Label(self.infoFrame, text = '유닛 개수')
            self.unit_label.pack(side='left', padx=self.pad_x)

            self.unit_size_entry = Entry(self.infoFrame)
            self.unit_size_entry.pack(side='left', padx=self.pad_x)

            self.activation_label = Label(self.infoFrame, text = '활성화 함수')
            self.activation_label.pack(side='left', padx=self.pad_x)

            self.activations = ['없음', 'relu', 'tanh', 'sigmoid', 'softmax']
            self.activation_cmb = ttk.Combobox(self.infoFrame, state='readonly', values = self.activations)
            self.activation_cmb.pack(side='left', padx=self.pad_x)

        if self.type == 'Conv2D':
            self.filter_label = Label(self.infoFrame, text = '필터 개수')
            self.filter_label.pack(side='left', padx=self.pad_x)

            self.filter_size_entry = Entry(self.infoFrame)
            self.filter_size_entry.pack(side='left', padx=self.pad_x)

            self.kernel_width_label = Label(self.infoFrame, text="커널 가로")
            self.kernel_width_label.pack(side="left", padx=self.pad_x)
            self.kernel_width_entry = Entry(self.infoFrame, width=5)
            self.kernel_width_entry.pack(side="left", padx=self.pad_x)
            self.kernel_height_label = Label(self.infoFrame, text="커널 세로")
            self.kernel_height_label.pack(side="left", padx=self.pad_x)
            self.kernel_height_entry = Entry(self.infoFrame, width=5)
            self.kernel_height_entry.pack(side="left", padx=self.pad_x)

            # self.stride_width_label = Label(self.infoFrame, text="커널 이동 간격(가로)")
            # self.stride_width_label.pack(side="left", padx=self.pad_x)
            # self.stride_width_entry = Entry(self.infoFrame, width=5)
            # self.stride_width_entry.pack(side="left", padx=self.pad_x)
            # self.stride_height_label = Label(self.infoFrame, text="커널 이동 간격(세로)")
            # self.stride_height_label.pack(side="left", padx=self.pad_x)
            # self.stride_height_entry = Entry(self.infoFrame, width=5)
            # self.stride_height_entry.pack(side="left", padx=self.pad_x)

            self.pooling_label = Label(self.infoFrame, text='데이터 풀링')
            self.pooling_label.pack(side='left', padx=self.pad_x)
            self.pooling_cmb = ttk.Combobox(self.infoFrame, state='readonly', values=['없음', 'Max Pooling', 'Mean Pooling'])
            self.pooling_cmb.pack(side='left', padx=self.pad_x)

            self.flatten_label = Label(self.infoFrame, text="데이터 펼치기(다음이 Dense)")
            self.flatten_label.pack(side="left", padx=self.pad_x)
            self.boolVar = BooleanVar()
            self.flatten_checkbox = Checkbutton(self.infoFrame, variable = self.boolVar)
            self.flatten_checkbox.pack(side="left", padx=self.pad_x)

        if self.type == 'RNN':
            self.unit_label = Label(self.infoFrame, text='유닛 개수')
            self.unit_label.pack(side='left', padx=self.pad_x)

            self.unit_size_entry = Entry(self.infoFrame)
            self.unit_size_entry.pack(side='left', padx=self.pad_x)

            self.keep_size_label = Label(self.infoFrame, text="timestep 유지(다음이 RNN)")
            self.keep_size_label.pack(side="left", padx=self.pad_x)

            self.boolVar = BooleanVar()
            self.keep_size_checkbox = Checkbutton(self.infoFrame, variable=self.boolVar)
            self.keep_size_checkbox.pack(side="left", padx=self.pad_x)


        if 'fixed_out' in kwargs and kwargs['fixed_out']:
            #print('fixed!')
            self.deleteable = False
            self.addable = False
            self.typeLabel.config(fg='red')

            #Dense
            self.unit_size_entry.insert(0, self.designer.output_shape)
            self.unit_size_entry.configure(state = 'disabled')
            if self.designer.model_type == '회귀':
                self.activation_cmb.current(0)
            elif self.designer.model_type == '분류':
                self.activation_cmb.current(4)
            self.activation_cmb.configure(state = 'disabled')


        if self.type == 'Output':
            self.deleteable = False
            self.addable = False
            self.typeLabel.config(fg='red')
            self.input_size_label = Label(self.infoFrame, text = f'출력 크기 : {self.designer.output_shape}')
            self.input_size_label.pack(side="left", padx=self.pad_x)





        self.menu = Menu(self, tearoff=0)
        self.menu.add_command(label="삭제", command = lambda : self.designer.deleteLayer(self))
        self.menu.entryconfig("삭제", state="normal" if self.deleteable else "disabled")
        self.typeMenu = Menu(self.menu, tearoff=0)
        self.typeMenu.add_command(label='Dense', command = lambda : self.designer.addLayer(self, 'Dense'))
        self.typeMenu.add_command(label='Conv2D', command = lambda : self.designer.addLayer(self, 'Conv2D'))
        self.typeMenu.add_command(label='RNN', command = lambda : self.designer.addLayer(self, 'RNN'))
        #self.typeMenu.add_command(label='LSTM', command = lambda : self.designer.addLayer(self, 'LSTM'))
        self.menu.add_cascade(label='다음 레이어 추가', menu = self.typeMenu, state="normal" if self.addable else "disabled")

        self.typeLabel.bind("<Button-3>", self.popup_menu)



    def popup_menu(self, event):
        #print('!!')
        self.menu.tk_popup(event.x_root, event.y_root)


class ScrollableFrame(tk.LabelFrame):
    def __init__(self, parent):
        self.parent = parent
        self.width = parent.width - 400
        self.height = parent.height - 100

        super().__init__(parent, width=self.width, height = self.height, text ="모델 설계 - 우클릭으로 삭제 혹은 다음 레이어 추가 (빨간색은 설정에 따른 기본 구조로 삭제 불가능)")


        self.canvas = Canvas(self, width=self.width, height=self.height)
        #self.canvas.grid(row = 0, column = 0, stickynsew')
        self.canvas.pack(side = "left", fill="both", expand = True)

        self.yScrollBar = ttk.Scrollbar(self, orient=VERTICAL, command=self.canvas.yview)
        #self.yScrollBar.grid(row=0, column = 1, sticky='ns')
        self.yScrollBar.pack(side="right", fill='y')

        self.win = tk.Frame(self.canvas, background='white', width=self.width, height=self.height)
        self.canvas.create_window(self.width//2, 0, window=self.win, anchor = 'n')

        self.canvas.config(yscrollcommand = self.yScrollBar.set, scrollregion=(0, 0, self.width, self.height))

        #self.yScrollBar.lift(self.win)



class ModelUser(tkinter.Toplevel):
    def __init__(self, parent, model_path):
        super().__init__(parent)

        self.parent = parent
        self.width = 1000
        self.height = 600
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        self.geometry(
            f"{self.width}x{self.height}+{int(self.screen_width / 2 - self.width / 2)}+{int(self.screen_height / 2 - self.height / 2)}")
        self.protocol("WM_DELETE_WINDOW", self.closing)

        self.model_path = model_path
        self.model_name = os.path.basename(os.path.dirname(model_path))
        self.model_folder_path = os.path.dirname(model_path)
        self.model = tensorflow.keras.models.load_model(model_path)

        with open(os.path.join(os.path.dirname(model_path), f'{self.model_name}.pickle'), 'rb') as f:
            self.model_base_data = pickle.load(f)

        self.input_data = None
        self.target_data = None

        #####################
        self.modeFrame = LabelFrame(self, text="모드 선택")
        self.modeFrame.grid(row = 0, column=0, pady=5)

        self.train_text = Label(self.modeFrame, text = "체크 시 훈련 / 해제 시 예측")
        self.train_text.pack(side='left', padx=5)
        # self.train_bool = BooleanVar()
        # self.train_checkbox = Checkbutton(self.modeFrame, variable = self.train_bool, command = self.change_targetframe)
        # self.train_checkbox.pack(side='left', padx=5)
        self.runtype_cmb = ttk.Combobox(self.modeFrame, state='readonly', values=['훈련', '테스트', '예측'])
        self.runtype_cmb.pack(side='left', padx=5)
        self.runtype_cmb.bind("<<ComboboxSelected>>", lambda event: self.change_frames())

        ###############################

        self.dataFrame = LabelFrame(self, text = "데이터 준비")
        self.dataFrame.grid(row=1, column=0, pady=5)
        self.inputFrame = Frame(self.dataFrame)
        self.inputFrame.pack(fill='x')
        self.input_data_text = Label(self.inputFrame, text = "입력 데이터(numpy)")
        self.input_data_text.pack(side='left', padx=5)
        self.input_file_entry = Entry(self.inputFrame, state='disabled')
        self.input_file_entry.pack(side='left', padx=5)
        self.input_select_btn = Button(self.inputFrame, text = "찾아보기", command = lambda: self.set_data(self.input_file_entry, True))
        self.input_select_btn.pack(side='left', padx=5)

        self.targetFrame = Frame(self.dataFrame)
        self.targetFrame.pack(fill='x')
        #############################
        self.configFrame = LabelFrame(self, text="학습 설정")
        self.configFrame.grid(row=2, column=0, pady=5)
        self.loss_text = Label(self.configFrame, text='손실 함수')
        self.loss_text.pack(side='left', padx=5)
        self.loss_cmb = ttk.Combobox(self.configFrame, state='readonly', values=['mse(회귀)', 'binary-crossentropy(이진 분류)', 'categorical_crossentropy(다중 분류)'], width=25)
        self.loss_cmb.pack(side='left', padx=5)

        self.metrics_text = Label(self.configFrame, text='평가 지표')
        self.metrics_text.pack(side='left', padx=5)
        self.metrics_cmb = ttk.Combobox(self.configFrame, state='readonly', values=['mae(회귀)', 'accuracy(분류)'], width=25)
        self.metrics_cmb.pack(side='left', padx=5)

        self.batch_size_text = Label(self.configFrame, text='배치(입력단위) 크기')
        self.batch_size_text.pack(side='left', padx=5)
        self.batch_size_entry = Entry(self.configFrame, width=10)
        self.batch_size_entry.pack(side='left', padx=5)

        self.epoch_size_text = Label(self.configFrame, text='학습 횟수', width=10)
        self.epoch_size_text.pack(side='left', padx=5)
        self.epoch_size_entry = Entry(self.configFrame)
        self.epoch_size_entry.pack(side='left', padx=5)


        self.lossdict={'mse(회귀)':'mse', 'binary-crossentropy(이진 분류)':'binary_crossentropy', 'categorical_crossentropy(다중 분류)':'categorical_crossentropy'}
        self.metricsdict={'mae(회귀)':['mae'], 'accuracy(분류)':['accuracy']}

        ##############3
        self.runbtn = Button(self, text='시작', command=self.run_model)
        self.runbtn.grid(row = 3, column=0, pady=10)

        #################################
        self.logFrame = LabelFrame(self, text = '실행 로그')
        self.logFrame.grid(row = 4, column = 0, sticky='ew', padx=5)

        self.yscrollbar = Scrollbar(self.logFrame)
        self.yscrollbar.pack(side="right", fill='y')
        self.xscrollbar = Scrollbar(self.logFrame)
        self.xscrollbar.pack(side="bottom", fill='x')

        self.logText = Text(self.logFrame, xscrollcommand=self.xscrollbar.set, yscrollcommand=self.yscrollbar.set)
        self.logText.pack(side="left", fill='both', expand=True)

        self.xscrollbar.configure(command = self.logText.xview)
        self.yscrollbar.configure(command = self.logText.yview)

    def closing(self):
        self.parent.attributes("-disabled", False)
        self.destroy()


    def set_data(self, entry:Entry, isInput):
        f = tkinter.filedialog.askopenfilename(title = '데이터 선택', filetypes = (("numpy file", "*.npy"),("numpy file", "*.npz")), initialdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'), parent = self)
        if len(f) > 0:
            entry.configure(state = 'normal')
            entry.delete(0, END)
            entry.insert(0, os.path.basename(f))
            entry.configure(state = 'disabled')

            if isInput:
                self.input_data = np.load(f)
            else:
                self.target_data = np.load(f)

    def change_frames(self):
        #print(self.runtype_cmb.get())
        self.input_data = None
        self.target_data = None
        self.input_file_entry.configure(state='normal')
        self.input_file_entry.delete(0, END)
        self.input_file_entry.configure(state='disabled')
        for w in self.targetFrame.winfo_children():
            w.destroy()
        if self.runtype_cmb.get() in ['훈련', '테스트']:
            target_data_text = Label(self.targetFrame, text="타겟 데이터(numpy)")
            target_data_text.pack(side='left', padx=5)
            target_file_entry = Entry(self.targetFrame, state='disabled')
            target_file_entry.pack(side='left', padx=5)
            target_select_btn = Button(self.targetFrame, text="찾아보기", command=lambda: self.set_data(target_file_entry, False))
            target_select_btn.pack(side='left', padx=5)



        if self.runtype_cmb.get() == '훈련':
            self.loss_cmb.configure(state='readonly')
            self.loss_cmb.set('')
            self.metrics_cmb.configure(state='readonly')
            self.metrics_cmb.set('')
            self.batch_size_entry.configure(state='normal')
            self.batch_size_entry.delete(0,END)
            self.epoch_size_entry.configure(state='normal')
            self.epoch_size_entry.delete(0, END)

        elif self.runtype_cmb.get() == '테스트':
            self.loss_cmb.set('')
            self.loss_cmb.configure(state='disabled')
            self.metrics_cmb.set('')
            self.metrics_cmb.configure(state='disabled')
            self.batch_size_entry.configure(state='normal')
            self.batch_size_entry.delete(0, END)
            self.epoch_size_entry.delete(0, END)
            self.epoch_size_entry.configure(state='disabled')

        else:
            self.loss_cmb.set('')
            self.loss_cmb.configure(state='disabled')
            self.metrics_cmb.set('')
            self.metrics_cmb.configure(state='disabled')
            self.batch_size_entry.configure(state='normal')
            self.batch_size_entry.delete(0, END)
            self.epoch_size_entry.delete(0, END)
            self.epoch_size_entry.configure(state='disabled')



    def run_model(self):
        if self.runtype_cmb.get()=='훈련':
            self.model.compile(loss = self.lossdict[self.loss_cmb.get()], metrics = self.metricsdict[self.metrics_cmb.get()])
            #self.model.summary()
            original_stdout = sys.stdout
            sys.stdout = StdoutChanger(textbox=self.logText)
            print('Train Start')
            history = self.model.fit(x=self.input_data, y=self.target_data, batch_size = int(self.batch_size_entry.get()), epochs = int(self.epoch_size_entry.get()), validation_split = 0.2, verbose = 2)
            print('Train Done')
            print('\n')
            now = datetime.datetime.now().strftime('%Y.%m.%d_%H-%M-%S')
            sys.stdout = original_stdout
            tensorflow.keras.models.save_model(self.model,  f'{self.model_folder_path}/{self.model_name}_{now}.h5')
            tkinter.messagebox.showinfo(title='완료', message='모델 학습 및 저장이 완료되었습니다.', parent=self)

        elif self.runtype_cmb.get()=='테스트':
            # self.model.summary()
            original_stdout = sys.stdout
            sys.stdout = StdoutChanger(textbox=self.logText)
            print('Test Start')
            history = self.model.evaluate(x=self.input_data, y=self.target_data, batch_size=int(self.batch_size_entry.get()), verbose=2)
            print('Test Done')
            print('\n')
            sys.stdout = original_stdout

        else:
            original_stdout = sys.stdout
            sys.stdout = StdoutChanger(textbox=self.logText)
            print('Predict Start')
            res = self.model.predict(x=self.input_data, batch_size = int(self.batch_size_entry.get()))
            print('Predict Done')
            print('\n')
            sys.stdout = original_stdout

            save_result_dir = tkinter.filedialog.asksaveasfilename(title = '데이터 선택', filetypes = (("numpy file", "*.npy"),),
                                                                   initialdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'),
                                                                   defaultextension = '.npy', parent = self)
            with open(save_result_dir, 'wb') as temp:
                np.save(temp, res)

            tkinter.messagebox.showinfo(title='완료', message='모델 예측 데이터가 저장되었습니다.', parent=self)


class StdoutChanger(object):
    def __init__(self, textbox):
        self.text_space = textbox

    def write(self, string):
        self.text_space.insert(END, string)

    def flush(self):
        pass

if __name__ == '__main__':
    app = SimpleDL()
    app.iconbitmap(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'res/S-DL.ico'))
    app.mainloop()




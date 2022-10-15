#Wang Liguo
#Tsinghua University
from distutils.command.clean import clean
from glob import glob
from inspect import isclass
import os
from socket import ntohl
from matplotlib.colors import cnames
import pandas as pd
import numpy as np
import math
import sys
import re
import subprocess
import threading
from fnmatch import fnmatch
import matplotlib.pyplot as plt 
from matplotlib import font_manager as fm
from matplotlib import cm

class newstring(str):
    def __init__(self,value):
        str.__init__(value)
        self.value = value
    def split(self):
        c_list = []
        tmp = ''
        feature = ' ()|\n'
        for c in self.value:
            if not c in feature:
                tmp = tmp + c
            else:
                if len(tmp)>0:
                    c_list.append(tmp)
                    tmp = ''
        final_list = []
        old_elem = ''
        for elem in c_list:
            if elem=='with':
                final_list.append(old_elem+' with DH')
                continue
            if elem=='DH':
                continue
            final_list.append(elem)
            old_elem = elem
        return final_list

def readindata(filename):
    text = []
    with open(filename) as f:
        for ln in f.readlines():
            text.append(newstring(ln))
    index = []
    lens = 0
    for i in range(1,len(text)):
        if text[i][0:2]=='--':
            break
        lens = lens + 1
    data = np.zeros([lens,len(text[0].split())-1])
    for i in range(1,len(text)):
        if text[i][0:2]=='--':
            break
        index.append(text[i].split()[0])
        for j in range(1,len(text[i].split())):
            if text[i].split()[j] == '|':
                data[i-1][j-1] = np.nan
            else:
                data[i-1][j-1]=float(text[i].split()[j])
    columns = text[0].split()[1:]
    dataframe = pd.DataFrame(data=data,index=index,columns=columns)
    return dataframe

def plot_binding_bar(work_dir,dataframe):
    '''Plot the bar figure from total MMPBSA data'''
    names = [('Binding Free Energy\nBinding = MM + PB + SA',
             ['Binding','MM','PB','SA']),
             ('Molecule Mechanics\nMM = COU + VDW',
             ['MM','COU','VDW']),
             ('Poisson Boltzman\nPB = PBcom - PBpro - PBlig',
             ['PB','PBcom','PBpro','PBlig']),
             ('Surface Area\nSA = SAcom - SApro - SAlig',
             ['SA','SAcom','SApro','SAlig'])]
    fig,axs = plt.subplots(2,2,figsize=(8,8),dpi=72)
    axs = np.ravel(axs)

    for ax,(title,name) in zip(axs,names):
        ax.bar(name,dataframe[name].mean(),width=0.5,
               yerr=dataframe[name].std(),color=['r','g','b','y'])
        for i in range(len(dataframe[name].mean())):
            ax.text(name[i],dataframe[name].mean()[i],
                    '%.3f'%dataframe[name].mean()[i],
                    ha='center',va='center')
        ax.grid(b=True,axis='y')
        ax.set_xlabel('Energy Decomposition Term')
        ax.set_ylabel('Free energy (kJ/mol)')
        ax.set_title(title)
    plt.suptitle('MMPBSA Results')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(work_dir,'MMPBSA_Results.png'))
    plt.show()

def plot_plot_pie(work_dir,datas):
    '''Plot the composition curve and pie figure'''
    fig,axs = plt.subplots(2,2,figsize=(8,8),dpi=72)
    axs = np.ravel(axs)

    names = [('Composition of MMPBSA',[0,1,4]),
             ('Composition of MM',[1,2,3]),
             ('Composition of PBSA',[4,5,6])]
    labels = ['res_MMPBSA','resMM','resMM_COU','resMM_VDW',
             'resPBSA','resPBSA_PB','resPBSA_SA']
    colors = ['black','blue','red']
    linestyles = ['-','--',':']
    alphas = [1,0.4,0.4]
    for ax,(title,name) in zip(axs[:-1],names):
        for i in range(len(name)):
            ax.plot(range(datas[name[i]].shape[1]),datas[name[i]].mean(),
                    color=colors[i],alpha=alphas[i],label=labels[name[i]],
                    linestyle=linestyles[i],linewidth=2.5)
        ax.grid(b=True,axis='y')
        ax.set_xlabel('Residues No.')
        ax.set_ylabel('Free Energy Contribution (kJ/mol)')
        ax.legend(loc='best')
        ax.set_title(title)
    
    explode = np.zeros([datas[0].shape[1]])
    maxposition = np.where(datas[0].mean() == datas[0].mean().abs().max())
    maxposition = np.append(maxposition,np.where(datas[0].mean() == 
                            -1 * datas[0].mean().abs().max()))
    explode[maxposition] = 0.4
    colors = cm.rainbow(np.arange(datas[0].shape[1])/datas[0].shape[1])
    patches, texts, autotexts = axs[-1].pie(abs(datas[0].mean()/datas[0].mean().sum()*100),
                explode=explode,labels=datas[0].columns,autopct='%1.1f%%',
                colors=colors,shadow=True,startangle=90,labeldistance=1.1,
                pctdistance=0.8)
    axs[-1].axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle
    axs[-1].set_title('Composition of MMPBSA')
    # set font size
    proptease = fm.FontProperties()
    proptease.set_size('xx-small')
    # font size include: xx-small,x-small,small,medium,large,x-large.xx-large or numbers
    plt.setp(autotexts,fontproperties=proptease)
    plt.setp(texts,fontproperties=proptease)
    plt.suptitle('MMPBSA Energy Composition')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(work_dir,'MMPBSA_Energy_Composition.png'),dpi=600)
    plt.show()

def DH_split(ori_data:pd.DataFrame):
    data = pd.DataFrame()
    data_with_dh = pd.DataFrame()
    for columnname, column in ori_data.iteritems():
        if 'with DH' in columnname:
            data_with_dh[columnname.strip(' with DH')] = column
        else:
            data[columnname] = column
    for columnname, column in data.iteritems():
        if not columnname in data_with_dh.columns:
            data_with_dh[columnname] = column
    return data, data_with_dh

def get_frame(filename:str):
    gmx_run = os.popen('{0} check -f {1} 2>&1'.format(gmx,filename))
    gmx_out = gmx_run.readlines()
    flag = False
    for line in gmx_out:
        txt = line.split()
        if ('Item' in txt) and ('#frames' in txt) and ('Timestep' in txt):
            flag = True
        if flag and ('Step' in txt):
            return txt

def count_ele(txt:list,pattern:str):
    count = 0
    for line in txt:
        words = line.split()
        for word in words:
            if word == pattern:
                count = count + 1
    return count

def show_logo():
    with open('gmx_mmpbsa_logo.dat','r') as f:
      print(f.read())

def set_default():
################################################################################
# 设置运行环境, 计算参数
# setting up environmets and parameters
################################################################################
    global tasks 
    tasks= 1
    global frames
    frames=10000000
    global trj
    trj='traj.xtc'		# 轨迹文件 trajectory file
    global tpr
    tpr='topol.tpr'		# tpr文件  tpr file
    global ndx
    ndx='index.ndx'		# 索引文件 index file
    global pro
    pro='Protein'			# 蛋白索引组   index group name of protein
    global lig
    lig='Ligand'			# 配体索引组   index group name of ligand
    global step
    step=0	# 从第几步开始运行 step number to run
            # 1. 预处理轨迹: 复合物完整化, 团簇化, 居中叠合, 然后生成pdb文件
            # 2. 获取每个原子的电荷, 半径, LJ参数, 然后生成qrv文件
            # 3. MM-PBSA计算: pdb->pqr, 输出apbs, 计算MM, APBS
            # 1. pre-processe trajectory, whole, cluster, center, fit, then generate pdb file
            # 2. abstract atomic parameters, charge, radius, C6/C12, then generate qrv file
            # 3. run MM-PBSA, pdb->pqr, apbs, then calculate MM, PB, SA
    global gmx
    gmx='gmx'								# /path/to/GMX/bin/gmx_mpi
    global dump
    dump="{0} dump".format(gmx)		# gmx dump
    global trjconv
    trjconv="{0} trjconv -dt 25000".format(gmx)			# gmx trjconv, use -b -e -dt, NOT -skip
    global apbs
    apbs='apbs'				# APBS(Linux)
    os.system('export MCSH_HOME=/dev/null')				# APBS io.mc
    global pid
    pid='pid'				# 输出文件($$可免重复) prefix of the output files($$)
    global err
    err='_{0}.err'.format(pid)		# 屏幕输出文件 file to save the message from the screen
    global qrv
    qrv='_{0}.qrv'.format(pid)		# 电荷/半径/VDW参数文件 to save charge/radius/vdw parmeters
    global pdb
    pdb='_{0}.pdb'.format(pid)		# 轨迹构型文件 to save trajectory
    global radType
    radType=1			# 原子半径类型 radius of atoms (0:ff; 1:mBondi)
    global radLJ0
    radLJ0=1.2			# 用于LJ参数原子的默认半径(A, 主要为H) radius when LJ=0 (H)
    global meshType
    meshType=0			# 网格大小 mesh (0:global  1:local)
    global gridType
    gridType=1			# 格点大小 grid (0:GMXPBSA 1:psize)
    global cfac
    cfac=3				# 分子尺寸到粗略格点的放大系数
    global fadd                    # Factor to expand mol-dim to get coarse grid dim
    fadd=10				# 分子尺寸到细密格点的增加值(A)
    global df                    # Amount added to mol-dim to get fine grid dim (A)
    df=0.5				# 细密格点间距(A) The desired fine mesh spacing (A)
    global isClean
    isClean=0

    # 极性计算设置(Polar)
    global PBEset
    PBEset='\n\
    temp  298.15      # 温度\n\
    pdie  2           # 溶质介电常数\n\
    sdie  78.54       # 溶剂介电常数, 真空1, 水78.54\n\
    \n\
    npbe              # PB方程求解方法, lpbe(线性), npbe(非线性), smbpe(大小修正)\n\
    bcfl  mdh         # 粗略格点PB方程的边界条件, zero, sdh/mdh(single/multiple Debye-Huckel), focus, map\n\
    srfm  smol        # 构建介质和离子边界的模型, mol(分子表面), smol(平滑分子表面), spl2/4(三次样条/7阶多项式)\n\
    chgm  spl4        # 电荷映射到格点的方法, spl0/2/4, 三线性插值, 立方/四次B样条离散\n\
    swin  0.3         # 立方样条的窗口值, 仅用于 srfm=spl2/4\n\
    \n\
    srad  1.4         # 溶剂探测半径\n\
    sdens 10          # 表面密度, 每A^2的格点数, (srad=0)或(srfm=spl2/4)时不使用\n\
    \n\
    ion charge  1 conc 0.15 radius 0.95  # 阳离子的电荷, 浓度, 半径\n\
    ion charge -1 conc 0.15 radius 1.81  # 阴离子\n\
    \n\
    calcforce  no\n\
    calcenergy comps'

    # 非极性计算设置(Apolar/Non-polar)
    global PBAset
    PBAset='\n\
    temp  298.15 # 温度\n\
    srfm  sacc   # 构建溶剂相关表面或体积的模型\n\
    swin  0.3    # 立方样条窗口(A), 用于定义样条表面\n\
    \n\
    # SASA\n\
    srad  1.4    # 探测半径(A)\n\
    gamma 1      # 表面张力(kJ/mol-A^2)\n\
    \n\
    #gamma const 0.0301248 0         # AMBER-PB4 .0072*cal2J 表面张力, 常数\n\
    #gamma const 0.0226778 3.84928\n\
    #gamma const 0.027     0\n\
    \n\
    press  0     # 压力(kJ/mol-A^3)\n\
    bconc  0     # 溶剂本体密度(A^3)\n\
    sdens 10\n\
    dpos  0.2\n\
    grid  0.1 0.1 0.1\n\
    \n\
    # SAV\n\
    #srad  1.29      # SAV探测半径(A)\n\
    #press 0.234304  # 压力(kJ/mol-A^3)\n\
    \n\
    # WCA\n\
    #srad   1.25           # 探测半径(A)\n\
    #sdens  200            # 表面的格点密度(1/A)\n\
    #dpos   0.05           # 表面积导数的计算步长\n\
    #bconc  0.033428       # 溶剂本体密度(A^3)\n\
    #grid   0.45 0.45 0.45 # 算体积分时的格点间距(A)\n\
    \n\
    calcforce no\n\
    calcenergy total'

def check_gmx_apbs():
################################################################################
# 检查 gmx, apbs 是否可以运行
# check gmx, apbs availability
################################################################################
    str=os.popen('{0} --version | grep -i "GROMACS version"'.format(gmx))
    if (step<3) and (len(str.read())==0):
        raise Exception("!!! ERROR !!!  GROMACS NOT available !\n")
    str=os.popen('{0} --version 2>&1 | grep -i "Poisson-Boltzmann"'.format(apbs))
    if len(str.read())==0:
        raise Exception("!!! WARNING !!!  APBS NOT available !\n")

def parse_command(parse:list):
    global useDH, useTS, isCAS,trj,tpr,ndx,pro,lig,cas,tasks,isClean
    useDH,useTS,isCAS=0,0,0
    cas=''
    it = iter(parse)
    while True:
        try:
            item = next(it)
            if item=='-f':
                trj = next(it)
                continue
            if item=='-s':
                tpr = next(it)
                continue
            if item=='-n':
                ndx = next(it)
                continue
            if item=='-pro':
                pro = next(it)
                continue
            if item=='-lig':
                lig = next(it)
                continue
            if item=='-nt':
                tasks = int(float(next(it)))
                continue
            if item=='-cou':
                useDH = 1
                continue
            if item=='-ts':
                useTS = 1
                continue
            if item=='-cas':
                isCAS = 1
                continue
            if item=='-clean':
                isClean= 1
        except StopIteration:
            break
    if isCAS:
        it = iter(parse)
        while True:
            try:
                item = next(it)  
                if item=='-cas':
                    while True:
                        item = next(it)
                        if not item in['-s','-f','-n','-pro','-lig','-cou','-ts','-cas','-nt','-clean']:
                            cas = cas + item
                        else:
                            break
            except StopIteration:
                break
    if len(lig)==0 or lig=='none':
        global withLig,com
        withLig=0
        com=pro
        lig=pro
    else:
        withLig=1
        com=pro+"_"+lig
    if step<3:
################################################################################
# 检查所需文件, 索引组
# check needed files, index group
################################################################################
        if not os.path.exists(trj):
            raise Exception("!!! ERROR !!! trajectory File NOT Exist !\n")
        if not os.path.exists(tpr):
            raise Exception("!!! ERROR !!! topology File NOT Exist !\n")
        if not os.path.exists(ndx):      
            raise Exception("!!! ERROR !!! index File NOT Exist !\n")  
        ndx_f = open(ndx,'r')
        ndx_content = ndx_f.readlines()
        ndx_f.close()
        count_lig = count_ele(ndx_content,lig)
        count_pro = count_ele(ndx_content,pro)
        if count_pro==0:
            raise Exception("!!! ERROR !!! [ {0} ] NOT in $ndx !\n".format(pro)) 
        if count_lig==0:
            raise Exception("!!! ERROR !!! [ {0} ] NOT in $ndx !\n".format(lig)) 
        if count_pro>1:
            raise Exception("!!! ERROR !!! more than ONE [ {0} ] in $ndx !\n".format(pro)) 
        if count_lig>1:
            raise Exception("!!! ERROR !!! more than ONE [ {1} ] in $ndx !\n".format(lig)) 
        if withLig:
            with open('gmx_mmpbsa_1.dat','r') as f:
                awf_f = f.read()
            names = ['trjconv','tpr','ndx','pro','lig','step','gmx','dump','trj','apbs','pid','err','qrv','pdb',\
            'radType','radLJ0','meshType','gridType','cfac','fadd','df','PBEset','PBAset','useDH','useTS','isCAS',\
            'cas','withLig','com']
            keys = [trjconv,tpr,ndx,pro,lig,step,gmx,dump,trj,apbs,pid,err,qrv,pdb,\
            radType,radLJ0,meshType,gridType,cfac,fadd,df,PBEset,PBAset,useDH,useTS,isCAS,\
            cas,withLig,com]
            count = 0
            for item in names:
                c = '{0}="{1}"\n'.format(item,keys[count])
                awf_f = c + awf_f
                count = count + 1
            p = subprocess.Popen(awf_f,shell=True,executable="/bin/bash")
            p.wait()
            ndx = '_{0}.ndx'.format(pid)

def preprocess():
################################################################################
# 1. 预处理轨迹: 复合物完整化, 团簇化, 居中叠合, 然后生成pdb文件
#    请检查pdb文件确保构型PBC处理正确
# 1. pre-processe trajectory, whole, cluster, center, fit, then generate pdb file
################################################################################
    if step<=1:
        global trjwho,trjcnt,trjcls,frames,timestep
        trjwho = '_{0}~trj_who.xtc'.format(pid)
        trjcnt = '_{0}~trj_cnt.xtc'.format(pid)
        trjcls = '_{0}~trj_cls.xtc'.format(pid)
        p = subprocess.Popen('echo {0} | {1}  -s {2} -n {3} -f {4} -o {5} -pbc whole &>>{6}'.format(\
            com,trjconv,tpr,ndx,trj,trjwho,err),shell=True,executable="/bin/bash")
        p.wait()
        if not os.path.exists(trjwho):
            raise Exception("!!! ERROR !!! gmx trjconv Failed ! Check {0} to find out why.\n".format(err))
        if withLig:
            # usful for single protein and ligand
            p = subprocess.Popen('echo "{0}" | {1}  -s {2} -n {3} -f {4} -o {5} -pbc mol -center &>>{6}'.format(\
            lig+'\n'+com,trjconv,tpr,ndx,trjwho,trjcnt,err),shell=True,executable="/bin/bash")
            p.wait()
            p = subprocess.Popen('echo "{0}" | {1}  -s {2} -n {3} -f {4} -o {5} -pbc cluster &>>{6}'.format(\
            com+'\n'+com,trjconv,tpr,ndx,trjcnt,trjcls,err),shell=True,executable="/bin/bash")
            p.wait()
            txt = get_frame(trjcls)
            frames = int(float(txt[1]))-1
            timestep = int(float(txt[2]))
            taskTimepoint = []
            s = -1
            e = -1
            global pdb_enu
            for _ in range(tasks):
                s = e + 1
                e = s + int(frames/tasks*timestep)
                if e > frames*timestep:
                    e = frames*timestep
                taskTimepoint.append([s,e])
            pdb_enu = []
            count = 0
            for task in taskTimepoint:
                pdb_task = '_{0}{1}.pdb'.format(pid,count)
                pdb_enu.append(pdb_task)
                p = subprocess.Popen('echo "{0}" | {1}  -s {2} -n {3} -f {4} -o {5} -b {6} -e {7} -fit rot+trans  &>>{8} '.format(\
                lig+'\n'+com,trjconv,tpr,ndx,trjcls,pdb_task,task[0],task[1],err),shell=True,executable="/bin/bash")
                p.wait()
                count = count + 1
        else:
            p = subprocess.Popen('echo {0} | {1}  -s {2} -n {3} -f {4} -o {5} &>>{6} -pbc mol -center'.format(\
            lig+'\n'+com,trjconv,tpr,ndx,trjwho,pdb,err),shell=True,executable="/bin/bash")
            p.wait()

def produce_qrv(pdb_task):
################################################################################
# 2. 获取每个原子的电荷, 半径, LJ参数, 然后生成qrv文件
# 2. abstract atomic parameters, charge, radius, C6/C12, then generate qrv file
#    feel free to change radius with radType
#    radType=0: radius from C6/C12, or radLJ0 if either C6 or C12 is zero
#    radType=1: mBondi
################################################################################        
    if step<=2:
        global qrv,pdb,isCAS
        step2_f = open('gmx_mmpbsa_2.dat','r')
        step2_order = step2_f.read()
        step2_f.close()
        names = ['trjconv','tpr','ndx','pro','lig','step','gmx','dump','trj','apbs','pid','err','qrv','pdb',\
            'radType','radLJ0','meshType','gridType','cfac','fadd','df','PBEset','PBAset','useDH','useTS','isCAS',\
                'cas','withLig','com','trjwho','trjcnt','trjcls']
        keys = [trjconv,tpr,ndx,pro,lig,step,gmx,dump,trj,apbs,pid,err,qrv,pdb_task,\
            radType,radLJ0,meshType,gridType,cfac,fadd,df,PBEset,PBAset,useDH,useTS,isCAS,\
                cas,withLig,com,trjwho,trjcnt,trjcls]
        count = 0
        for item in names:
            c = '{0}="{1}"\n'.format(item,keys[count])
            step2_order = c + step2_order
            count = count + 1
        p = subprocess.Popen(step2_order,shell=True,executable='/bin/bash')
        p.wait()
        with open('test.log','w') as f:
            f.write(step2_order)
        if isCAS:
            qrv = "_{0}_CAS.qrv".format(pid)
            pdb = "_{0}_CAS.pdb".format(pid)

def cal_mmpbsa(_pid,pdb_task):
################################################################################
# 3. MM-PBSA计算: pdb->pqr, 输出apbs, 计算MM, APBS
# 3. run MM-PBSA, pdb->pqr, apbs, then calculate MM, PB, SA
################################################################################
    if step<=3:
        step3_f = open('gmx_mmpbsa_3.dat','r')
        step3_order = step3_f.read()
        step3_f.close()
        names = ['trjconv','tpr','ndx','pro','lig','step','gmx','dump','trj','apbs','pid','err','qrv','pdb',\
            'radType','radLJ0','meshType','gridType','cfac','fadd','df','PBEset','PBAset','useDH','useTS','isCAS',\
                'cas','withLig','com','trjwho','trjcnt','trjcls']
        keys = [trjconv,tpr,ndx,pro,lig,step,gmx,dump,trj,apbs,_pid,err,qrv,pdb_task,\
            radType,radLJ0,meshType,gridType,cfac,fadd,df,PBEset,PBAset,useDH,useTS,isCAS,\
                cas,withLig,com,trjwho,trjcnt,trjcls]
        count = 0
        for item in names:
            c = '{0}="{1}"\n'.format(item,keys[count])
            step3_order = c + step3_order
            count = count + 1
        order_file_name = "_{0}_order_{1}.bash".format(pid,pdb_task)
        with open(order_file_name ,'w') as f:
            f.write(step3_order)
        p = subprocess.Popen('chmod 777 {0}'.format(order_file_name),shell=True,executable='/bin/bash')
        p.wait()
        p = subprocess.Popen('./{0} >> err.log'.format(order_file_name),shell=True,executable='/bin/bash')
        p.wait()

class output_data():
    def __init__(self,suffix:str) -> None:
        self.suffix = suffix
        self.pid = []
        self.fileList = []
        self.fileHandle = []
    def complete_filename(self):
        for __pid__ in self.pid:
            self.fileList.append(__pid__+self.suffix)
    def merge(self):
        for file in self.fileList:
            f = open(file,'r')
            self.fileHandle.append(f)
        allData = []
        flag = True
        for f in self.fileHandle:
            if flag:
                allData = allData + f.readlines()
                flag = False
            else:
                allData = allData + f.readlines()[1:] 
        lines = ''
        for line in allData:
            lines = lines + line
        for file in self.fileHandle:
            file.close()
        return lines

def data_merge(task_num):
    suffix_list = ['~resPBSA.dat','~resPBSA_PB.dat','~resPBSA_SA.dat',\
    '~resMM.dat', '~resMM_COU.dat', '~res_MMPBSA.dat','~MMPBSA.dat','~resMM_VDW.dat']
    dataList = []
    for suffix in suffix_list:
        data = output_data(suffix)
        for task in range(task_num):
            data.pid.append('_{0}{1}'.format(pid,task))
        dataList.append(data)
    for data in dataList:
        data.complete_filename()
        txt = data.merge()
        with open('_'+pid+data.suffix,'w') as f:
            f.write(txt)

def rm_temp():
    global isClean,tasks,pid
    if isClean:
        rm_feature=['io.mc','_'+pid+'$'+'*','*bash']
        rm_list = []
        for f in rm_feature:
            if '$' in f:
                for i in range(tasks):
                    rm_list.append(f.replace('$',str(i)))
            else:
                rm_list.append(f)
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                for f in rm_list:
                    if fnmatch(file,f):
                        os.remove(os.path.join(root,file))
                        break

def data_format():
    prefix = "_" + pid
    files = ['MMPBSA','res_MMPBSA','resMM','resMM_COU','resMM_VDW',
             'resPBSA','resPBSA_PB','resPBSA_SA']
    datas = []
    for file in files:
        filename = prefix + '~' + file + '.dat'
        datas.append(readindata(filename))
    try:
        os.mkdir('data')
        os.mkdir('data_with_DH')
    except Exception:
        pass
    count = 0
    for data in datas:
        dat, dat_with_DH = DH_split(data)
        dat.to_csv('data/'+files[count]+'.csv')
        dat_with_DH.to_csv('data_with_DH/'+files[count]+'.csv')
        count = count + 1

# def data_analysis():
#     plot_binding_bar(datas_with_dh[0])
#     plot_plot_pie(datas_with_dh[1:])

if __name__ == '__main__':
    show_logo()
    set_default()
    check_gmx_apbs()
    parse_command(sys.argv)
    print('>> set up environmets and parameters: OK !\n')
    preprocess()
    print(">> preprocessing trajectory: OK !\n")
    global pdb_enu
    produce_qrv(pdb_enu[0])
    print('>> generate qrv file: OK !\n')
    threadSet = []
    count = 0
    for pdb_task in pdb_enu:
        l = threading.Thread(target=cal_mmpbsa,args=(pid+str(count),pdb_task))
        threadSet.append(l)
        count = count + 1
        l.start()
    for l in threadSet:
        l.join()
    print('>> Calculate MM PB SA: OK !\n')
    data_merge(tasks)
    print('>> Merge Data: OK !\n')
    data_format()
    print('>> Reformat Data: OK !\n')
    rm_temp()
    print('>> Remove tempoary file: OK !\n')
    # data_analysis()
    # print('>> Data analysis: OK !\n')


from glob import glob
from inspect import isclass
import os
from socket import ntohl
import pandas as pd
import numpy as np
import math
import sys
import re
import subprocess

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
    trjconv="{0} trjconv -dt 2500".format(gmx)			# gmx trjconv, use -b -e -dt, NOT -skip
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
    global useDH, useTS, isCAS,trj,tpr,ndx,pro,lig,cas,tasks
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
                        if not item in['-s','-f','-n','-pro','-lig','-cou','-ts','-cas','-nt']:
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
        print('>> 0. set up environmets and parameters: OK !\n')

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
        print(">> 1. preprocessing trajectory: OK !\n")

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
        print('>> 2. generate qrv file: OK !\n')
        if isCAS:
            qrv = "_{0}_CAS.qrv".format(pid)
            pdb = "_{0}_CAS.pdb".format(pid)

def cal_mmpbsa(pdb_task):
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
        keys = [trjconv,tpr,ndx,pro,lig,step,gmx,dump,trj,apbs,pid,err,qrv,pdb_task,\
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
        subprocess.run('nohup ./{0} >> err.log 2>&1 &'.format(order_file_name),shell=True,executable='/bin/bash')

def rm_temp():
    os.system(r'rm -f io.mc \#*')
       
                                               
if __name__ == '__main__':
    show_logo()
    set_default()
    check_gmx_apbs()
    parse_command(sys.argv)
    preprocess()
    global pdb_enu
    produce_qrv(pdb_enu[0])
    for pdb_task in pdb_enu:    
        cal_mmpbsa(pdb_task)
    rm_temp()

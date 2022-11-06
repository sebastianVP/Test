import numpy as np
import matplotlib.pyplot as plt
import argparse

PATH      = "C:/Users/soporte/Documents/"
'''
filename  = "simulacion001.bin" 
filename  = "data_2m_sawtooth.bin"
dir_file  = PATH+filename
fs        = 6*1e6#2*1e6
T         = 2*1e-3#1e-3
c         = 3*1e8
B         = 0.75*1e6
type_     = "sawtooth"

UPDATE: 
1) 
MEJORA EN FILTRO DE ARCHIVOS
Filtro para archivos con la extension.
Por ejemplo : .bin o  .hdf5
2)
MEJORA EN CONSOLA
Parametros de entrada por consola.
filename, fs,T,type of signal(sawtooth, triangle,sine), muestras de tiempo ploteo.
3)
MEJORA EN GNURADIO
Mejorar el grabado de datos
con el archivo que tenga informacion del CHIRP fs,T,B,Tipo
4)
LECTURA POR IPP
COMO SIGNAL CHAIN
5)
MEJORA CREANDO CLASES DENTRO DEL METODO MAIN
'''
def main(args):
    """
    Command example:
    python .\readerCHIRP.py "simulacion001.bin" -fs 2000000  --period_chirp 1e-3 --swip_chirp 750000 --waveform_type sawtooth
    python .\readerCHIRP.py "data_2m_sawtooth.bin" -fs 6000000  --period_chirp 2e-3 --swip_chirp 750000 --waveform_type sawtooth

    """
    filename = args.experiment
    fs       = args.sample_rate
    T        = args.period_chirp
    B        = args.swip_chirp
    type_    = args.waveform_type
    c        = 3*1e8

    print("\nSTARTING... [OK]\n")
    print("Brief Overview")
    print("EXPERIMENT INFORMATION")

    print("PATH             :", PATH, ", filename:",filename)
    dir_file  = PATH+filename
    data = np.fromfile(open(dir_file),dtype= np.float32)
    NFFT      = len(data)
    #---  VECTOR DE TIEMPO Y FRECUENCIA ------
    time = np.arange(len(data))/fs
    f    = np.linspace(-int(NFFT)/2,int(NFFT)/2 -1, int(NFFT))*fs/NFFT

    print("Length Data      : %d muestras."%len(data))
    print("Tiempo total     :", time[-1]*1000, "mseg.")
    print("fmax             :", np.max(f)/1000,"Khz.")
    #--- DOMINIO DE FRECUENCIA -MAGNITUD ---
    data_fft = np.fft.fftshift(np.fft.fft(data))
    D_F_fft  = np.abs(data_fft)/len(f)
    D   = D_F_fft[f>0]


    #--- TYPE OF SIGNAL AND PROCESS ---
    if type_ == "sawtooth":
        R_C = c/(2*(B/T))
    elif type_ == "triangle":
        R_C =c/(4*(B/T))
    else: 
        #SINE
        R_C = c/(12*(B/T))
    print("Tipo de Señal    : %s"%type_)
    print("Factor del BEAT  : %s"%R_C)
    print("Samples D_F      :", len(D_F_fft[f>0]))
    #--- RANGE ---

    delta_r = (f[f>0][1]-f[f>0][0])*R_C
    range_  = f[f>0]*R_C   
    print("Resolucion       : %1.4f m."%delta_r)
    print("MAX Range        :%1.5f Km."%(range_[-1]/1000.0))

    #--- PLOT ---
    fig ,axes = plt.subplots(3,1,figsize=(8,6))
    fig.tight_layout(pad=3.4)
    ax1,ax2,ax3 = axes
    # PLOTEO EN miliseg para el primer grafico.
    t_    = time[:int(len(data)/10)]*1000.0  # mseg
    data_ = data[:int(len(data)/10)]
    #t_    = time[:6000]/1000.0  # mseg
    #data_ = data[:6000]
    # PLOTEO Km para el tercer grafico.
    distance = range_/1000
    print("Muestras 1er grafico: ",len(t_))
    print("Tiempo              : ",t_[-1], "mseg")

    ax1.plot(t_,data_)
    ax2.plot(f[f>0]/1e5,D)
    ax3.plot(distance,D)
    fig.suptitle('EXPERIMENT %s'%filename ,color='black',fontsize=14)

    ax1.title.set_text('Time Beat Plot')
    ax1.set_ylabel("VOLTAGE")
    ax1.set_xlabel("mseg")
    ax2.title.set_text('Profile Frequency Domain Plot')
    ax2.set_xlabel("10**5 Hz")
    ax2.set_ylabel("MAGNITUD")
    ax3.title.set_text("Profile FFT-Range Plot ")
    ax3.set_ylabel("MAGNITUD")
    ax3.set_xlabel("Kilometros")
    #ax3.set_xlim(0,150*1e3)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to process SOPHY data")
    parser.add_argument('experiment',help='Experiment filename')
    parser.add_argument('-fs','--sample_rate'  ,default=1e6       ,type=float , help='Sample rate hz')
    parser.add_argument('--period_chirp',default=1e-3      ,type=float , help='Periodo de la señal Chirp')
    parser.add_argument('--swip_chirp'   ,default=1e3       ,type=float , help='Barrido Chirp')
    parser.add_argument('--waveform_type',default='sawtooth',type=str   ,  required=True)
    args  = parser.parse_args()
    main(args)


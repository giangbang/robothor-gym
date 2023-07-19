from setuptools import setup, find_packages


def start_xserver() -> None:
    with open("frame-buffer", "w") as writefile:
        writefile.write(
            """#taken from https://gist.github.com/jterrace/2911875
    XVFB=/usr/bin/Xvfb
    XVFBARGS=":1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset"
    PIDFILE=./frame-buffer.pid
    case "$1" in
    start)
        /sbin/start-stop-daemon --start --quiet --pidfile $PIDFILE --make-pidfile --background --exec $XVFB -- $XVFBARGS
        ;;
    stop)
        /sbin/start-stop-daemon --stop --quiet --pidfile $PIDFILE
        rm $PIDFILE
        ;;
    restart)
        $0 stop
        $0 start
        ;;
    *)
            exit 1
    esac
    exit 0
        """
        )

    os.system("apt-get install daemon >/dev/null 2>&1")

    os.system("apt-get install wget >/dev/null 2>&1")

    os.system(
        "wget http://ai2thor.allenai.org/ai2thor-colab/libxfont1_1.5.1-1ubuntu0.16.04.4_amd64.deb >/dev/null 2>&1"
    )

    os.system(
        "wget --output-document xvfb.deb http://ai2thor.allenai.org/ai2thor-colab/xvfb_1.18.4-0ubuntu0.12_amd64.deb >/dev/null 2>&1"
    )

    os.system("dpkg -i libxfont1_1.5.1-1ubuntu0.16.04.4_amd64.deb >/dev/null 2>&1")

    os.system("dpkg -i xvfb.deb >/dev/null 2>&1")

    os.system("rm libxfont1_1.5.1-1ubuntu0.16.04.4_amd64.deb")

    os.system("rm xvfb.deb")

    os.system("bash frame-buffer start")

    os.environ["DISPLAY"] = ":1"
    
    os.system("apt --fix-broken install")

    os.system("sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-utils")
    
    os.system("apt update & apt upgrade")

    os.system("sudo apt install pciutils")

    # os.system("sudo ai2thor-xorg start")

start_xserver()


setup(
    name = "robothor-gym",
    version = "0.0.1",
    description = ("Robothor environment for RL with gym API"),
    packages=find_packages(),
    install_requires=[
        "numpy",
        "ai2thor"
    ],
)

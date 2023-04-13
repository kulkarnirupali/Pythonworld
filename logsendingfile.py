import os
import time
import psutil
import urllib3
import smtplib
import schedule
from sys import *
from email import encoders
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def is_connected():
    try:
        urllib3.connection_from_url('http://216.58.192.142',Timeout = 1)
        return True
    except urllib3.connection_from_url as err:
        return False

def Mailsender(filename,time):
    try:
        fromaddr = "rmkulkarni17@gmail.com"
        toaddr = "rmkulkarni17@gmail.com"

        msg = MIMEMultipart()
        msg['from'] = fromaddr
        msg['To'] = toaddr

        body = """
        Hello ,
        welcome to Mavellous Infosystems.
        Please find attached document which contains Log of Running
        process..
        
        This is auto gennerated mail .
        
        Thanks & Regards,
        Kulkarni Rupali Manoj
        Marvellous Infosystem
        """ %(toaddr,time)

        Subject = """
        Marvellous Infosystem Process Log generated at :
        """%(time)
        msg['Subject'] = Subject
        msg.attach(MIMEText(body,'plain'))
        attachment = open(filename,"rb")
        p = MIMEBase('application','octet-stream')
        p.set_payload((attachment).read())
        encoders.encode_base64(p)
        p.add_header('Content - Diposition',"attachment;filename = " % filename)
        msg.attach(p)
        s = smtplib.SMTP('smtp.gmail.com',587)
        s.starttls()
        s.login(fromaddr,"...............")
        text = msg.as_string()
        s.sendmail(fromaddr,toaddr,text)
        s.quit()
        print("Log file successfully sent through Mail")

    except Exception as E:
        print("Unable to send the mail",E)

def Processlog(log_dir = "Marvellous"):
    listprocess = []

    if not os.path.exists(log_dir):
        try:
            os.mkdir(log_dir)
        except:
            pass

    separator = "-" * 80
    log_path = os.path.join(log_dir,"MarvellousLog.log"%(time.ctime()))
    f = open(log_path,'w')
    f.write(separator + "\n")
    f.write("MarvellouS Infosystem Process Logger :"+time.ctime() + "\n")
    f.write(separator + "\n")
    f.write("\n")


    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=['pid','name','username'])
            vms = proc.memory_info().vms / (1024 * 1024)
            pinfo['vms'] = vms
            listprocess.append(pinfo);

        except (psutil.NoSuchProcess,psutil.AccessDenied,psutil.ZombieProcess):
            pass

    for element in listprocess:
        f.write("\n" % element)

    connected = is_connected()

    if connected:
        starttime = time.time()
        Mailsender(log_path,time.ctime())
        endtime = time.time()

        print("Took seconds to send mails "%(endtime-starttime))
    else:
        print("THere is no internet connection")

def main():
    print("Appliaction name : "+argv[0])
    if (len(argv) != 2):
        print("Error : Invalid number of arguements")
        exit()

    if (argv[1] == "-h") or (argv[1] == "-H"):
        print("This script is used log record of Running processes")
        exit()

    if (argv[1] == "-u") or (argv[1] == "-U"):
        print("usage : Application name Absolutepath_of_Directory")
        exit()

    try:
        schedule.every(int(argv[1])).minutes.do(Processlog)
        while True:
            schedule.run_pending()
            time.sleep(1)

    except ValueError:
        print("Error : Invalid datatype of input")

    except Exception as E:
        print("Error : Invalid datatype of input",E)

if __name__ == "__main__":
    main()




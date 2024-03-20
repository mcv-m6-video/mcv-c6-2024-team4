from tracking import run

if __name__ == '__main__':
    #opt = parse_opt()
    #main(opt)
    config = './config.yaml'
    weight = None
    source = '/home/tda/Desktop/Master/C6/mcv-c6-2024-team4/data/aic19-track1-mtmc-train/train/S01/c001/vdo.avi'
    display = False
    save = False
    run(config, weight, source, display, save)
    
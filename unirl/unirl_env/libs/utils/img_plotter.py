import os, sys, json
from matplotlib import pyplot as plt
from shapely import wkt, geometry, affinity
from absl import app, flags, logging
import fiona
from fiona.crs import from_epsg

FLAGS = flags.FLAGS

flags.DEFINE_list('pnu', '', 'pnu')
flags.DEFINE_boolean('plot_history', False, '')


def load_data(path):
    with open(path, 'r', encoding="utf-8") as f:
        return json.load(f)

def plot(data, path):

    data['plot_geom'] = wkt.loads(data['plot_geom'])
    data['site_geom'] = wkt.loads(data['site_geom'])
    data['legal_geom'] = wkt.loads(data['legal_geom'])
    data['parking_geom'] = wkt.loads(data['parking_geom'])
    data['unit_geom'] = wkt.loads(data['unit_geom'])
    data['elev_geom'] = wkt.loads(data['elev_geom'])
    data['stair_geom'] = wkt.loads(data['stair_geom'])
    data['corridor_geom'] = wkt.loads(data['corridor_geom'])


    fig = plt.figure(figsize=(20,4))
    plt.clf()
    for i in range(len(data['unit_geom'])):
        ax = fig.add_subplot(1, len(data['unit_geom']), i+1, aspect='equal', anchor='S')
        ax.plot(*data['site_geom'].exterior.xy, color='grey')
        ax.plot(*data['legal_geom'][i].exterior.xy, color='red')
        [ax.plot(*x.exterior.xy, color='black') for x in data['unit_geom'][i]]
        if not data['elev_geom'].is_empty:
            ax.plot(*data['elev_geom'].exterior.xy, color='green')
        ax.plot(*data['stair_geom'].exterior.xy, color='green')
        ax.plot(*data['corridor_geom'][i].exterior.xy, color='blue')
        if i == 0:
            [ax.plot(*x.exterior.xy, color='purple') for x in data['parking_geom']]
    plt.savefig(path)


    # if FLAGS.save_img:
    #     os.makedirs(os.path.join(FLAGS.log_dir, 'img'), exist_ok=True)
    #     if not FLAGS.euid:
    #         plt.savefig(os.path.join(FLAGS.log_dir, 'img', 'wuid-{}.png'.format(FLAGS.wuid)))
    #     else:
    #         plt.savefig(os.path.join(FLAGS.log_dir, 'img', 'euid-{}.png'.format(FLAGS.euid)))
    # if FLAGS.show_img:
    #     plt.show()
    # plt.close()



    # if FLAGS.save_shp:

    #     if not FLAGS.euid:
    #         logdir = os.path.join(FLAGS.log_dir, 'shp', 'wuid-{}'.format(FLAGS.wuid))
    #     else:
    #         logdir = os.path.join(FLAGS.log_dir, 'shp', 'euid-{}'.format(FLAGS.euid))

    #     os.makedirs(logdir, exist_ok=True)

    #     data['elev_geom'] = [data['elev_geom'] for _ in data['unit_geom']]
    #     data['stair_geom'] = [data['stair_geom'] for _ in data['unit_geom']]
    #     data['plot_geom'] = {'info':[{'geom':translate_z(data['plot_geom'], 0), 'floor':1, 'idx':0}], 'type':'Polygon'}
    #     data['site_geom'] = {'info':[{'geom':translate_z(data['site_geom'], 0), 'floor':1, 'idx':0}], 'type':'Polygon'}
    #     data['parking_geom'] = {'info':[{'geom':x, 'floor':1, 'idx':i} for i, x in enumerate(translate_z(data['parking_geom'], 0))], 'type':'Polygon'}
    #     data['unit_geom'] = {'info':[[{'geom':x_, 'floor':f, 'idx':i} for i, x_ in enumerate(x)] for f, x in enumerate(translate_zs(data['unit_geom'], [f * 30 for f in range(len(data['unit_geom']))]))], 'type':'Polygon'}
    #     data['elev_geom'] = {'info':[{'geom':x, 'floor':f, 'idx':0} for f, x in enumerate(translate_zs(data['elev_geom'], [f * 30 for f in range(len(data['elev_geom']))]))], 'type':'Polygon'}
    #     data['stair_geom'] = {'info':[{'geom':x, 'floor':f, 'idx':0} for f, x in enumerate(translate_zs(data['stair_geom'], [f * 30 for f in range(len(data['stair_geom']))]))], 'type':'Polygon'}
    #     data['corridor_geom'] = {'info':[{'geom':x, 'floor':f, 'idx':0} for f, x in enumerate(translate_zs(data['corridor_geom'], [f * 30 for f in range(len(data['corridor_geom']))]))], 'type':'Polygon'}

    #     labels = ['plot_geom', 'site_geom', 'parking_geom', 'unit_geom', 'elev_geom', 'stair_geom', 'corridor_geom']

    #     if not FLAGS.euid:
    #         path = os.path.join(logdir, 'wuid-{}.shp'.format(FLAGS.wuid))
    #     else:
    #         path = os.path.join(logdir, 'euid-{}.shp'.format(FLAGS.euid))

    #     schema = {
    #         'geometry': 'Polygon',
    #         'properties': {
    #             'label': 'str',
    #             'floor': 'int',
    #             'idx': 'int'
    #         },
    #     }
    #     with fiona.open(path, mode='w', driver='ESRI Shapefile', schema=schema, crs=from_epsg(3857)) as c:
    #         for label in labels:
    #             write_label(c, path, data, label)
        
# def write_label(c, path, data, label):

#     infos = flatten_list(data[label]['info'])
    
#     for info in infos:
#         if not info['geom'].is_empty:
#             c.write({
#                 'geometry': geometry.mapping(info['geom']),
#                 'properties': {'label': label,'floor': info['floor'], 'idx': info['idx']},
#             })
    

# def flatten_list(lists):
#     if isinstance(lists, list):
#         if len(lists) < 1:
#             return lists
#         elif isinstance(lists[0], list):
#             return flatten_list(lists[0]) + flatten_list(lists[1:])
#         else:
#             return lists[:1] + flatten_list(lists[1:])
#     else:
#         return [lists]

# def translate_z(geom, z):
#     return affinity.translate(geom, xoff=z)
 
# def translate_zs(gs, zs):
#     return list(map(lambda x, y: translate_z(x, y), gs, zs))


def run():
    info_dir = os.path.join(FLAGS.log_dir, 'info', '_'.join(FLAGS.pnu))
    img_dir = os.path.join(FLAGS.log_dir, 'img', '_'.join(FLAGS.pnu))
    with os.scandir(info_dir) as entries:
        for entry in entries:
            info_uid_dir = entry
            img_uid_dir = os.path.join(img_dir, os.path.split(entry)[-1])

            with os.scandir(info_uid_dir) as json_files:
                for json_file in json_files:
                    if json_file.name.split('.')[-1] == 'json':
                        data = load_data(json_file)
                        os.makedirs(img_uid_dir, exist_ok=True)
                        plot(data, os.path.join(img_uid_dir, '.'.join([os.path.split(json_file)[-1].split('.')[0], 'png'])))

            if FLAGS.plot_history:
                info_uid_history_dir = os.path.join(info_uid_dir, 'history')
                img_uid_history_dir = os.path.join(img_uid_dir, 'history')
                with os.scandir(info_uid_history_dir) as json_files:
                    for json_file in json_files:
                        if json_file.name.split('.')[-1] == 'json':
                            data = load_data(json_file)
                            os.makedirs(img_uid_history_dir, exist_ok=True)
                            plot(data, os.path.join(img_uid_history_dir, '.'.join([os.path.split(json_file)[-1].split('.')[0], 'png'])))

def main(argv):

    
    if not FLAGS.log_dir:
        FLAGS.log_dir = './'
    

    run()

if __name__ == "__main__":
    
    app.run(main)


import theano
import numpy
import theano.tensor as tensor
from keras.layers import Embedding,GRU,Merge
from keras.callbacks import EarlyStopping
import datasource
#from model import VGG_19
from evaluation import t2i, i2t
from datasets import build_dictionary
from datasets import load_dataset
from theano.tensor.extra_ops import fill_diagonal
from keras.layers import Input, Dense
from keras.layers.core import Lambda, Masking
from keras.models import Model
from keras.utils.visualize_util import plot
from hyperopt import Trials, STATUS_OK, tpe,  fmin, tpe, hp, STATUS_OK, Trials


def compute_errors(s_emb, im_emb):
    """ Given sentence and image embeddings, compute the error matrix """
    erros = [order_violations(x, y) for x in s_emb for y in im_emb]
    return numpy.asarray(erros).reshape((len(s_emb), len(im_emb)))


def order_violations(s, im):
    """ Computes the order violations (Equation 2 in the paper) """
    return numpy.power(numpy.linalg.norm(numpy.maximum(0, s - im)),2)


def l2norm(X):
    """ Compute L2 norm, row-wise """
    norm = tensor.sqrt(tensor.pow(X, 2).sum(1))
    X /= norm[:, None]
    return X


def contrastive_loss(labels, predict):
    """For a minibatch of sentence and image embeddings, compute the pairwise contrastive loss"""
    global model_options
    margin = model_config['margin']
    res = theano.tensor.split(predict, [model_config['output_dim'], model_config['output_dim']], 2, axis=-1)
    s = res[0]
    im = res[1]
    im2 = im.dimshuffle(('x', 0, 1))
    s2 = s.dimshuffle((0, 'x', 1))
    errors = tensor.pow(tensor.maximum(0, im2 - s2), 2).sum(axis=2)
    diagonal = errors.diagonal()
    # compare every diagonal score to scores in its column (all contrastive images for each sentence)
    cost_s = tensor.maximum(0, margin - errors + diagonal)
    # all contrastive sentences for each image
    cost_im = tensor.maximum(0, margin - errors + diagonal.reshape((-1, 1)))
    cost_tot = cost_s + cost_im
    cost_tot = fill_diagonal(cost_tot, 0)
    return cost_tot.sum()


# main trainer
def train(params):

    try:

        #GRID SEARCH
        print (params)

        global  model_config
        model_config['margin'] = params['margin'] if 'margin' in params else model_config['margin']
        model_config['output_dim'] = params['output_dim'] if 'output_dim' in params else model_config['output_dim']
        model_config['max_cap_length'] = params['max_cap_length']  if 'max_cap_length' in params else model_config['max_cap_length']
        model_config['optimizer'] =  params['optimizer']  if 'optimizer' in params else model_config['optimizer'],
        model_config['dim_word'] = params['dim_word'] if 'dim_word' in params else model_config['dim_word']


        # Load training and development sets
        print ('Loading dataset')
        dataset = load_dataset(model_config['data'], cnn=model_config['cnn'])

        train = dataset['train']
        test = dataset['test']
        val = dataset['dev']

        # Create dictionary
        print ('Creating dictionary')

        worddict = build_dictionary(train['caps'] + val['caps'])
        print ('Dictionary size: ' + str(len(worddict)))
        model_config['worddict'] = len(worddict)


        print ('Loading data')
        train_iter = datasource.Datasource(train, batch_size=model_config['batch_size'], worddict=worddict)
        val_iter = datasource.Datasource(val, batch_size=model_config['batch_size'], worddict=worddict)
        test_iter = datasource.Datasource(test, batch_size=model_config['batch_size'], worddict=worddict)

        print ("Image model loading")
        # # this returns a tensor of emb_image
        image_input = Input(shape=(model_config['dim_cnn'],), name='image_input')
        X = Dense(model_config['output_dim'],)(image_input)
        X = Lambda(lambda x: l2norm(x))(X)
        emb_image = Lambda(lambda x: abs(x))(X)

        print ("Text model loading")
        # this returns a tensor of emb_cap
        cap_input = Input(shape=(model_config['max_cap_length'],), dtype='int32', name='cap_input')
        X = Masking(mask_value=0,input_shape=(model_config['max_cap_length'], model_config['output_dim']))(cap_input)
        X = Embedding(output_dim=model_config['dim_word'], input_dim=model_config['worddict'], input_length=model_config['max_cap_length'])(cap_input)
        X = GRU(output_dim=model_config['output_dim'], return_sequences=False)(X)
        X = Lambda(lambda x: l2norm(x))(X)
        emb_cap = Lambda(lambda x: abs(x))(X)

        print ("loading the joined model")
        merged = Merge( mode='concat')([emb_cap, emb_image])
        model = Model(input=[cap_input, image_input], output=[merged])

        print ("compiling the model")
        model.compile(optimizer=model_config['optimizer'][0], loss=contrastive_loss)

        # uncomment for model selection and add  validation_data=(gen_val_data()) when calling fit_generator
        # def gen_val_data():
        #     val_bacthes = [[x, im] for x, im in val_iter]
        #     x1 = []
        #     x2 = []
        #     for batch in val_bacthes:
        #         x1.append(batch[0])
        #         x2.append(batch[1])
        #     mat_x1 = numpy.array(x1).reshape(7*model_config['batch_size'],model_config['max_cap_length'])
        #     mat_x2 = numpy.array(x2).reshape(7*model_config['batch_size'], model_config['dim_cnn'])
        #     dummy = numpy.zeros(shape=(len(mat_x1), model_config['output_dim'] * 2))
        #     return [mat_x1,mat_x2], dummy
        #

        def train_generator(batch_size):
            def gen(batch_size):
                batches = [[x, im] for x, im in train_iter]
                dummy = numpy.zeros(shape=(batch_size, model_config['output_dim'] * 2))
                for batch in batches:
                    yield (batch, dummy)
            return gen


        #uncomment for model selection and add  callbacks=[early_stopping] when calling fit_generator
        #ModelCheckpoint('/home/igor/PycharmProjects/GRU/models', monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
        #early_stopping = EarlyStopping(monitor='val_loss', patience=50)

        train_hist = model.fit_generator(train_generator(batch_size=model_config['batch_size']),
                                         samples_per_epoch=(
                                             model_config['worddict'] / model_config['batch_size'] * model_config[
                                                 'batch_size']),
                                         nb_epoch= model_config['epoch'], verbose=2, class_weight=None, max_q_size=0)


        model.save_weights('my_model_weights.h5')

        print(train_hist.history)

        # uncomment in order to load model weights
        #model.load_weights('my_model_weights.h5')

        def eval_model():
            print ('evaluating model...')
            weights = model.get_weights()
            emb_w = weights[0]
            im_w = weights[1]
            im_b = weights[2]
            gru_weights = weights[3:12]

            test_model_im = Model(input=image_input, output=emb_image)
            test_model_im.set_weights([im_w, im_b])
            test_model_im.compile(optimizer='adam', loss=contrastive_loss)
            test_model_cap = Model(input=cap_input, output=emb_cap)
            test_model_cap.set_weights([emb_w]+ gru_weights)
            test_model_cap.compile(optimizer='adam', loss=contrastive_loss)

            test_cap, test_im = test_iter.all()
            all_caps = numpy.zeros(shape=(len(test_cap),model_config['max_cap_length']))
            all_images = numpy.zeros(shape=(len(test_cap), model_config['dim_cnn']))
            pred_cap = test_model_cap.predict(test_cap)
            pred_im = test_model_im.predict(test_im)
            test_errs = compute_errors(pred_cap, pred_im)

            r10_c, rmean_c = t2i(test_errs)
            r10_i, rmean_i = i2t(test_errs)
            print ("Image to text: %.1f %.1f" % (r10_i, rmean_i))
            print ("Text to image: %.1f %.1f" % (r10_c, rmean_c))


        #evaluate model - recall@10 & mean_rank metric
        eval_model()

        # uncomment for model selection
        #return {'loss': train_hist.history['loss'][0], 'status': STATUS_OK, 'model': model}

    except:
        raise





#Grid search configs

#best setting
# {'margin': 0.1, 'output_dim': 1024, 'optimizer': 'adam', 'dim_word': 500}
#Image to text: 85.6,
#Text to image: 75.7,


# uncomment for model selection
# space = { 'margin' : hp.choice('margin', [0.05, 0.1, 0.15]),
#           # 'batch_size': hp.choice('batch_size', [256, 128]),
#           'optimizer': hp.choice('optimizer', ['adam']),
#           'output_dim' : hp.choice('output_dim', [1024, 2048]),
#           'dim_word' : hp.choice('dim_word', [100, 300, 500]),
#
# }



model_config = {}

def trainer(config):
    global model_config
    model_config = config
    train(model_config)

    # uncomment for model selection
    #trials = Trials()
    #best = fmin(train, space, algo=tpe.suggest, max_evals=100, trials=trials)
    #print 'best: '
    #print best

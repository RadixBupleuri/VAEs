from random import randint
from data.dataloaders.bar_dataset import *
from data.dataloaders.bar_dataset_helpers import *
from MeasureVAE import MeasureVAE
from train_MeasureVAE import MeasureTrainer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt


class VAETester(object):
    def __init__(
            self,
            dataset,
            model: MeasureVAE,
            has_reg_loss=False,
            reg_type=None,
            reg_dim=0
    ):
        self.dataset = dataset
        self.model = model
        self.model.eval()
        self.has_reg_loss = has_reg_loss
        self.trainer_config = ''
        if self.has_reg_loss:
            if reg_type is not None:
                self.reg_type = reg_type
            self.reg_dim = reg_dim
            if self.reg_type == 'joint':
                self.reg_dim = [0, 1]
            if self.reg_type == 'joint_rhycomp_noterange':
                self.reg_dim = [0, 2]
            if self.reg_type == 'four_metrics':
                self.reg_dim = [0, 1, 2, 3]
            self.trainer_config = '[' + self.reg_type + ',' + str(self.reg_dim) + ']'
            self.model.update_trainer_config(self.trainer_config)
            self.model.load()
            self.model.cuda()
        self.filepath = os.path.join('models/',
                                     self.model.__repr__())

        self.decoder = self.model.decoder
        # freeze decoder
        self.train = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.z_dim = self.decoder.z_dim
        self.batch_size = 1  # PREVIOUSLY IT WAS 1, 256 for testing
        self.measure_seq_len = 24  # TODO: remove this hardcoding
        self.dir_path = os.path.dirname(os.path.realpath(__file__))

    def test_interpretability(self, batch_size, attr_type):
        """
        Tests the interpretability of the latent space for a partcular attribute
        :param batch_size: int, number of datapoints in mini-batch
        :param attr_type: str, attribute type
        :return: tuple(int, float): index of dimension with highest mutual info, interpretability score
        """
        (_, gen_val, gen_test) = self.dataset.data_loaders(
            batch_size=batch_size,
            split=(0.01, 0.01)
        )

        # compute latent vectors and attribute values
        z_all = []
        attr_all = []
        for sample_id, (score_tensor, metadata_tensor) in tqdm(enumerate(gen_test)):
            if isinstance(self.dataset, FolkNBarDataset):
                batch_size = score_tensor.size(0)
                score_tensor = score_tensor.view(batch_size, self.dataset.n_bars, -1)
                score_tensor = score_tensor.view(batch_size * self.dataset.n_bars, -1)
                metadata_tensor = metadata_tensor.view(batch_size, self.dataset.n_bars, -1)
                metadata_tensor = metadata_tensor.view(batch_size * self.dataset.n_bars, -1)
            # convert input to torch Variables
            score_tensor, metadata_tensor = (
                to_cuda_variable_long(score_tensor),
                to_cuda_variable_long(metadata_tensor)
            )
            # compute encoder forward pass
            z_dist = self.model.encoder(score_tensor)
            # sample from distribution
            z_tilde = z_dist.rsample()

            # compute attributes
            if attr_type == 'rhy_complexity':
                attr = self.dataset.get_rhy_complexity(score_tensor)
            elif attr_type == 'num_notes':
                attr = self.dataset.get_notes_density_in_measure(score_tensor)
            elif attr_type == 'note_range':
                attr = self.dataset.get_note_range_of_measure(score_tensor)
            elif attr_type == 'average_interval_jump':
                attr = self.dataset.get_average_pitch_interval_of_measure(score_tensor)
            z_all.append(to_numpy(z_tilde.cpu()))
            attr_all.append(to_numpy(attr.cpu()))
        z_all = np.concatenate(z_all)
        attr_all = np.concatenate(attr_all)

        # compute mutual information
        mutual_info = np.zeros(self.z_dim)
        for i in tqdm(range(self.z_dim)):
            mutual_info[i] = mutual_info_score(z_all[:, i], attr_all)
        dim = np.argmax(mutual_info)
        max_mutual_info = np.max(mutual_info)

        reg = LinearRegression().fit(z_all[:, dim:dim + 1], attr_all)
        score = reg.score(z_all[:, dim:dim + 1], attr_all)
        return dim, score

    def test_model(self, batch_size):
        """
        Runs the model on the test set
        :param batch_size: int, number of datapoints in mini-batch
        :return: tuple: mean_loss, mean_accuracy
        """
        (_, gen_val, gen_test) = self.dataset.data_loaders(
            batch_size=batch_size,
            split=(0.01, 0.01)
        )
        print('Num Test Batches: ', len(gen_test))
        mean_loss_test, mean_accuracy_test = self.loss_and_acc_test(gen_test)
        print('Test Epoch:')
        print(
            '\tTest Loss: ', mean_loss_test, '\n'
                                             '\tTest Accuracy: ', mean_accuracy_test * 100
        )

    def test_interp(self):
        """
        Tests the interpolation capabilities of the latent space
        :return: None
        """
        (_, gen_val, gen_test) = self.dataset.data_loaders(
            batch_size=1,  # TODO: remove this hard coding
            split=(0.01, 0.5)
        )
        gen_it_test = gen_test.__iter__()
        for _ in range(randint(0, len(gen_test))):
            tensor_score1, _ = next(gen_it_test)

        gen_it_val = gen_val.__iter__()
        for _ in range(randint(0, len(gen_val))):
            tensor_score2, _ = next(gen_it_val)

        tensor_score1 = to_cuda_variable(tensor_score1.long())
        tensor_score2 = to_cuda_variable(tensor_score2.long())
        self.test_interpolation(tensor_score1, tensor_score2, 10)

    def test_interpolation(self, tensor_score1, tensor_score2, n=1):
        """
        Tests the interpolation in the latent space for two random points in the
        validation and test set
        :param tensor_score1: torch tensor, (1, measure_seq_len)
        :param tensor_score2: torch tensor, (1, measure_seq_len)
        :param n: int, number of points for interpolation
        :return:
        """
        z_dist1 = self.model.encoder(tensor_score1)
        z_dist2 = self.model.encoder(tensor_score2)
        z1 = z_dist1.loc
        z2 = z_dist2.loc
        tensor_score = self.decode_mid_point(z1, z2, n)
        # tensor_score = torch.cat((tensor_score1, tensor_score, tensor_score2), 1)
        score = self.dataset.tensor_to_m21score(tensor_score.cpu())
        score.show()
        return score

    def decode_mid_point(self, z1, z2, n):
        """
        Decodes the mid-point of two latent vectors
        :param z1: torch tensor, (1, self.z_dim)
        :param z2: torch tensor, (1, self.z_dim)
        :param n: int, number of points for interpolation
        :return: torch tensor, (1, (n+2) * measure_seq_len)
        """
        assert (n >= 1 and isinstance(n, int))
        # compute the score_tensors for z1 and z2
        dummy_score_tensor = to_cuda_variable(torch.zeros(self.batch_size, self.measure_seq_len))
        _, sam1 = self.decoder(z1, dummy_score_tensor, self.train)
        _, sam2 = self.decoder(z2, dummy_score_tensor, self.train)
        # find the interpolation points and run through decoder
        tensor_score = sam1
        for i in range(n):
            z_interp = z1 + (z2 - z1) * (i + 1) / (n + 1)
            _, sam_interp = self.decoder(z_interp, dummy_score_tensor, self.train)
            tensor_score = torch.cat((tensor_score, sam_interp), 1)
        tensor_score = torch.cat((tensor_score, sam2), 1).view(1, -1)
        # score = self.dataset.tensor_to_score(tensor_score.cpu())
        return tensor_score

    def test_attr_reg_interpolations(self, num_points=10, dim=0, num_interps=20):
        for i in range(num_points):
            z = torch.randn(1, self.model.latent_space_dim)
            z1 = z.clone()
            z2 = z.clone()
            z1[:, dim] = -3.
            z2[:, dim] = 3.
            z1 = to_cuda_variable(z1)
            z2 = to_cuda_variable(z2)
            tensor_score = self.decode_mid_point(z1, z2, num_interps)
            score = self.dataset.tensor_to_m21score(tensor_score.cpu())
            score.show()

    def loss_and_acc_test(self, data_loader):
        """
        Computes loss and accuracy for test data
        :param data_loader: torch data loader object
        :return: (float, float)
        """
        mean_loss = 0
        mean_accuracy = 0

        for sample_id, (score_tensor, metadata_tensor) in tqdm(enumerate(data_loader)):
            if isinstance(self.dataset, FolkNBarDataset):
                batch_size = score_tensor.size(0)
                score_tensor = score_tensor.view(batch_size, self.dataset.n_bars, -1)
                score_tensor = score_tensor.view(batch_size * self.dataset.n_bars, -1)
                metadata_tensor = metadata_tensor.view(batch_size, self.dataset.n_bars, -1)
                metadata_tensor = metadata_tensor.view(batch_size * self.dataset.n_bars, -1)
            # convert input to torch Variables
            score_tensor, metadata_tensor = (
                to_cuda_variable_long(score_tensor),
                to_cuda_variable_long(metadata_tensor)
            )
            # compute forward pass
            weights, samples, _, _, _, _ = self.model(
                measure_score_tensor=score_tensor,
                measure_metadata_tensor=metadata_tensor,
                train=False
            )

            # compute loss
            recons_loss = MeasureTrainer.mean_crossentropy_loss(
                weights=weights,
                targets=score_tensor
            )
            loss = recons_loss
            # compute mean loss and accuracy
            mean_loss += to_numpy(loss.mean())
            accuracy = MeasureTrainer.mean_accuracy(
                weights=weights,
                targets=score_tensor
            )
            mean_accuracy += to_numpy(accuracy)
        mean_loss /= len(data_loader)
        mean_accuracy /= len(data_loader)
        return (
            mean_loss,
            mean_accuracy
        )

    def _plot_data_attr_dist(self, gen_test, dim1, dim2, reg_type):
        z_all = []
        attr_all = []
        for sample_id, (score_tensor, metadata_tensor) in tqdm(enumerate(gen_test)):
            if isinstance(self.dataset, FolkNBarDataset):
                batch_size = score_tensor.size(0)
                score_tensor = score_tensor.view(batch_size, self.dataset.n_bars, -1)
                score_tensor = score_tensor.view(batch_size * self.dataset.n_bars, -1)
                metadata_tensor = metadata_tensor.view(batch_size, self.dataset.n_bars, -1)
                metadata_tensor = metadata_tensor.view(batch_size * self.dataset.n_bars, -1)
            # convert input to torch Variables
            score_tensor, metadata_tensor = (
                to_cuda_variable_long(score_tensor),
                to_cuda_variable_long(metadata_tensor)
            )
            # compute encoder forward pass
            z_dist = self.model.encoder(score_tensor)
            # sample from distribution
            z_tilde = z_dist.rsample()

            # compute attributes
            if reg_type == 'rhy_complexity':
                attr = self.dataset.get_rhy_complexity(score_tensor)
            elif reg_type == 'num_notes':
                attr = self.dataset.get_notes_density_in_measure(score_tensor)
            elif reg_type == 'note_range':
                attr = self.dataset.get_note_range_of_measure(score_tensor)
            z_all.append(z_tilde)
            attr_all.append(attr)
        z_all = to_numpy(torch.cat(z_all, 0))
        attr_all = to_numpy(torch.cat(attr_all, 0))
        if self.trainer_config == '':
            reg_str = '[no_reg]'
        else:
            reg_str = self.trainer_config
        filename = self.dir_path + '/plots/' + reg_str + 'data_dist_' + reg_type + '_[' \
                   + str(dim1) + ',' + str(dim2) + '].png'
        self.plot_dim(z_all, attr_all, filename, dim1=dim1, dim2=dim2, xlim=6, ylim=6)

    def plot_data_attr_dist(self, dim1=0, dim2=1):
        """
        Plots the data distribution
        :param dim1: int,
        :param dim2: int,
        :return:
        """
        (_, _, gen_test) = self.dataset.data_loaders(
            batch_size=16,  # TODO: remove this hard coding
            split=(0.7, 0.15)
        )
        print('Num Test Batches: ', len(gen_test))
        self._plot_data_attr_dist(gen_test, dim1, dim2, 'rhy_complexity')
        self._plot_data_attr_dist(gen_test, dim1, dim2, 'num_notes')
        self._plot_data_attr_dist(gen_test, dim1, dim2, 'note_range')


    def plot_attribute_dist(self, attribute='num_notes', plt_type='pca'):
        """
        Plots the distribution of a particular attribute in the latent space
        :param attribute: str,
                num_notes, note_range, rhy_entropy, beat_strength, rhy_complexity
        :param plt_type: str, 'tsne' or 'pca'
        :return:
        """
        (_, _, gen_test) = self.dataset.data_loaders(
            batch_size=64,  # TODO: remove this hard coding
            split=(0.01, 0.01)
        )
        z_all = []
        n_all = []
        num_samples = 5
        for sample_id, (score_tensor, _) in tqdm(enumerate(gen_test)):
            # convert input to torch Variables
            if isinstance(self.dataset, FolkNBarDataset):
                batch_size = score_tensor.size(0)
                score_tensor = score_tensor.view(batch_size, self.dataset.n_bars, -1)
                score_tensor = score_tensor.view(batch_size * self.dataset.n_bars, -1)
            score_tensor = to_cuda_variable_long(score_tensor)
            # compute encoder forward pass
            z_dist = self.model.encoder(score_tensor)
            z_tilde = z_dist.loc
            z_all.append(z_tilde)
            if attribute == 'num_notes':
                attr = self.dataset.get_notes_density_in_measure(score_tensor)
            elif attribute == 'note_range':
                attr = self.dataset.get_note_range_of_measure(score_tensor)
            elif attribute == 'rhy_entropy':
                attr = self.dataset.get_rhythmic_entropy(score_tensor)
            elif attribute == 'beat_strength':
                attr = self.dataset.get_beat_strength(score_tensor)
            elif attribute == 'rhy_complexity':
                attr = self.dataset.get_rhy_complexity(score_tensor)
            else:
                raise ValueError('Invalid attribute type')
            for i in range(attr.size(0)):
                tensor_score = score_tensor[i, :]
                start_idx = self.dataset.note2index_dicts[START_SYMBOL]
                end_idx = self.dataset.note2index_dicts[END_SYMBOL]
                if tensor_score[0] == start_idx:
                    attr[i] = -0.1
                elif tensor_score[0] == end_idx:
                    attr[i] = -0.2
            n_all.append(attr)
            if sample_id == num_samples:
                break
        z_all = torch.cat(z_all, 0)
        n_all = torch.cat(n_all, 0)
        z_all = to_numpy(z_all)
        n_all = to_numpy(n_all)

        filename = self.dir_path + '/plots/' + plt_type + '_' + attribute + '_' + \
                   str(num_samples) + '_measure_vae.png'
        if plt_type == 'pca':
            self.plot_pca(z_all, n_all, filename)
        elif plt_type == 'tsne':
            self.plot_tsne(z_all, n_all, filename)
        elif plt_type == 'dim':
            self.plot_dim(z_all, n_all, filename)
        else:
            raise ValueError('Invalid plot type')

    def plot_transposition_points(self, plt_type='pca'):
        """
        Plots a t-SNE plot for data-points comprising of transposed measures
        :param plt_type: str, 'tsne' or 'pca'
        :return:
        """
        filepaths = self.dataset.valid_filepaths
        idx = random.randint(0, len(filepaths))
        original_score = get_music21_score_from_path(filepaths[idx])
        possible_transpositions = self.dataset.all_transposition_intervals(original_score)
        z_all = []
        n_all = []
        n = 0
        for trans_int in possible_transpositions:
            score_tensor = self.dataset.get_transposed_tensor(
                original_score,
                trans_int
            )
            score_tensor = self.dataset.split_tensor_to_bars(score_tensor)
            score_tensor = to_cuda_variable_long(score_tensor)
            z_dist = self.model.encoder(score_tensor)
            z_tilde = z_dist.loc
            z_all.append(z_tilde)
            t = np.arange(0, z_tilde.size(0))
            n_all.append(torch.from_numpy(t))
            # n_all.append(torch.ones(z_tilde.size(0)) * n)
            n += 1
        print(n)
        z_all = torch.cat(z_all, 0)
        n_all = torch.cat(n_all, 0)
        z_all = to_numpy(z_all)
        n_all = to_numpy(n_all)

        filename = self.dir_path + '/plots/' + plt_type + '_transposition_measure_vae.png'
        if plt_type == 'pca':
            self.plot_pca(z_all, n_all, filename)
        elif plt_type == 'tsne':
            self.plot_tsne(z_all, n_all, filename)
        else:
            raise ValueError('Invalid plot type')

    @staticmethod
    def plot_pca(data, target, filename):
        pca = PCA(n_components=2, whiten=False)
        pca.fit(data)
        pca_z = pca.transform(data)
        plt.scatter(
            x=pca_z[:, 0],
            y=pca_z[:, 1],
            c=target,
            cmap='viridis',
            alpha=0.3
        )
        plt.colorbar()
        plt.savefig(filename, format='png', dpi=300)
        plt.show()
        plt.close()

    @staticmethod
    def plot_tsne(data, target, filename):
        tsne = TSNE(n_components=2, verbose=1., perplexity=40, n_iter=300)
        tsne_z = tsne.fit_transform(data)
        plt.scatter(
            x=tsne_z[:, 0],
            y=tsne_z[:, 1],
            c=target,
            cmap="viridis",
            alpha=0.3
        )
        plt.colorbar()
        plt.savefig(filename, format='png', dpi=300)
        plt.show()
        plt.close()

    @staticmethod
    def plot_dim(data, target, filename, dim1=0, dim2=1, xlim=None, ylim=None):
        if xlim is not None:
            plt.xlim(-xlim, xlim)
        if ylim is not None:
            plt.ylim(-ylim, ylim)
        plt.scatter(
            x=data[:, dim1],
            y=data[:, dim2],
            c=target,
            s=12,
            linewidths=0,
            cmap="viridis",
            alpha=0.5
        )
        plt.colorbar()
        plt.savefig(filename, format='png', dpi=300)
        # plt.show()
        plt.close()
        print('saved: ' + filename)

    @staticmethod
    def get_cmap(n, name='hsv'):
        return plt.cm.get_cmap(name, n)


@click.command()
@click.option('--note_embedding_dim', default=10,
              help='size of the note embeddings')
@click.option('--metadata_embedding_dim', default=2,
              help='size of the metadata embeddings')
@click.option('--num_encoder_layers', default=2,
              help='number of layers in encoder RNN')
@click.option('--encoder_hidden_size', default=512,
              help='hidden size of the encoder RNN')
@click.option('--encoder_dropout_prob', default=0.5,
              help='float, amount of dropout prob between encoder RNN layers')
@click.option('--has_metadata', default=False,
              help='bool, True if data contains metadata')
@click.option('--latent_space_dim', default=256,
              help='int, dimension of latent space parameters')
@click.option('--num_decoder_layers', default=2,
              help='int, number of layers in decoder RNN')
@click.option('--decoder_hidden_size', default=512,
              help='int, hidden size of the decoder RNN')
@click.option('--decoder_dropout_prob', default=0.5,
              help='float, amount got dropout prob between decoder RNN layers')
@click.option('--batch_size', default=256,
              help='training batch size')
@click.option('--num_epochs', default=30, # 30 EPOCHS
              help='number of training epochs')
@click.option('--train/--test', default=False, # TRAIN
              help='train or test the specified model')
@click.option('--plot/--no_plot', default=True,
              help='plot the training log')
@click.option('--log/--no_log', default=True,
              help='log the results for tensorboard')
@click.option('--reg_loss/--no_reg_loss', default=True, # YES REG LOSS
              help='train with regularization loss')
@click.option('--reg_type', default='four_metrics', # REG TYPE FOUR METRICS
              help='attribute name string to be used for regularization')
@click.option('--reg_dim', default=0, # REG DIMS, overwritten in vae_trainer
              help='dimension along with regularization is to be carried out')
@click.option('--attr_plot/--no_attr_plot', default=True,
              help='if True plots the attribute dsitributions, else produces interpolations')

def run(note_embedding_dim,
         metadata_embedding_dim,
         num_encoder_layers,
         encoder_hidden_size,
         encoder_dropout_prob,
         latent_space_dim,
         num_decoder_layers,
         decoder_hidden_size,
         decoder_dropout_prob,
         has_metadata,
         batch_size,
         num_epochs,
         train,
         plot,
         log,
         reg_loss,
         reg_type,
         reg_dim,
         attr_plot
         ):

    is_short = False
    num_bars = 1
    folk_dataset_train = FolkNBarDataset(
        dataset_type='train',
        is_short=is_short,
        num_bars=num_bars)
    folk_dataset_test = FolkNBarDataset(
        dataset_type='test',
        is_short=is_short,
        num_bars=num_bars
    )

    model = MeasureVAE(
        dataset=folk_dataset_train,
        note_embedding_dim=note_embedding_dim,
        metadata_embedding_dim=metadata_embedding_dim,
        num_encoder_layers=num_encoder_layers,
        encoder_hidden_size=encoder_hidden_size,
        encoder_dropout_prob=encoder_dropout_prob,
        latent_space_dim=latent_space_dim,
        num_decoder_layers=num_decoder_layers,
        decoder_hidden_size=decoder_hidden_size,
        decoder_dropout_prob=decoder_dropout_prob,
        has_metadata=has_metadata
    )

    model.load()
    model.cuda()
    model.eval()

    folk_dataset_train.data_loaders(
        batch_size=batch_size,
        split=(0.70, 0.20)
    )

    (generator_train,
     generator_val,
     generator_test) = folk_dataset_train.data_loaders(
        batch_size=batch_size,
        split=(0.70, 0.20)
    )

    # z vector
    z0_actual_values = []
    z1_actual_values = []
    z2_actual_values = []
    z3_actual_values = []

    for sample_id, (score_tensor, metadata_tensor) in enumerate(generator_train):

        if isinstance(folk_dataset_train, FolkNBarDataset):
            batch_size = score_tensor.size(0)
            score_tensor = score_tensor.view(batch_size, folk_dataset_train.n_bars, -1)
            score_tensor = score_tensor.view(batch_size * folk_dataset_train.n_bars, -1)
            metadata_tensor = metadata_tensor.view(batch_size, folk_dataset_train.n_bars, -1)
            metadata_tensor = metadata_tensor.view(batch_size * folk_dataset_train.n_bars, -1)

        # convert input to torch Variables
        score_tensor, metadata_tensor = (
            to_cuda_variable_long(score_tensor),
            to_cuda_variable_long(metadata_tensor)
        )

        # compute encoder forward pass
        z_dist = model.encoder(score_tensor)
        # sample from distribution
        z_tilde = z_dist.rsample()

        for z_tilde_index in range(z_tilde.size()[0]):
            dim_0_val = z_tilde[z_tilde_index, 0].cpu().detach().numpy()
            dim_1_val = z_tilde[z_tilde_index, 1].cpu().detach().numpy()
            dim_2_val = z_tilde[z_tilde_index, 2].cpu().detach().numpy()
            dim_3_val = z_tilde[z_tilde_index, 3].cpu().detach().numpy()

            z0_actual_values.append(dim_0_val.tolist())
            z1_actual_values.append(dim_1_val.tolist())
            z2_actual_values.append(dim_2_val.tolist())
            z3_actual_values.append(dim_3_val.tolist())



if __name__ == "__main__":
    run()
from torch import distributions

from utils_trainer import Trainer
from MeasureVAE import MeasureVAE
from data.dataloaders.bar_dataset import *
import click

from MeasureVAE import MeasureVAE
from data.dataloaders.bar_dataset import *
from utils import *
import torch
import tqdm
class MeasureTrainer(Trainer):
    def __init__(
        self, dataset,
        model: MeasureVAE,
        lr=1e-4,
        has_reg_loss=False,
        reg_type=None,
        reg_dim=0,
    ):
        super(MeasureTrainer, self).__init__(dataset, model, lr)
        self.beta = 0.001
        self.cur_epoch_num = 0
        self.has_reg_loss = has_reg_loss
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
        self.warm_up_epochs = 10
        # self.scheduler = torch.optim.lr_scheduler.StepLR(
        #    optimizer=self.optimizer,
        #    step_size=30,
        #    gamma=0.1
        # )

    def loss_and_acc_for_batch(self, batch, epoch_num=None, train=True):
        """
        Computes the loss and accuracy for the batch
        Must return (loss, accuracy) as a tuple, accuracy can be None
        :param batch: torch Variable,
        :param epoch_num: int, used to change training schedule
        :param train: bool, True is backward pass is to be performed
        :return: scalar loss value, scalar accuracy value
        """
        if self.cur_epoch_num != epoch_num:
            flag = True
            self.cur_epoch_num = epoch_num
        else:
            flag = False
        # extract data
        score, metadata = batch
        # perform forward pass of model
        weights, samples, z_dist, prior_dist, z_tilde, z_prior = self.model(
            measure_score_tensor=score,
            measure_metadata_tensor=metadata,
            train=train
        )
        # compute loss
        recons_loss = self.mean_crossentropy_loss(weights=weights, targets=score)
        # dist_loss = self.compute_mmd_loss(z_tilde, z_prior)
        dist_loss = self.compute_kld_loss(z_dist, prior_dist)
        loss = recons_loss + dist_loss
        # compute accuracy
        accuracy = self.mean_accuracy(weights=weights,
                                      targets=score)
        if self.has_reg_loss:
            reg_loss = self.compute_reg_loss(z_tilde, score)
            loss += reg_loss
            if flag:
                print(recons_loss.item(), dist_loss.item(), reg_loss.item())
        else:
            if flag:
                print(recons_loss.item(), dist_loss.item())
        return loss, accuracy

    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return: tuple of Torch Variable objects
        """
        score_tensor, metadata_tensor = batch
        if isinstance(self.dataset, FolkNBarDataset):
            batch_size = score_tensor.size(0)
            score_tensor = score_tensor.view(batch_size, self.dataset.n_bars, -1)
            score_tensor = score_tensor.view(batch_size * self.dataset.n_bars, -1)
            metadata_tensor = metadata_tensor.view(batch_size, self.dataset.n_bars, -1)
            metadata_tensor = metadata_tensor.view(batch_size * self.dataset.n_bars, -1)
        # convert input to torch Variables
        batch_data = (
            to_cuda_variable_long(score_tensor),
            to_cuda_variable_long(metadata_tensor)
        )
        return batch_data

    def update_scheduler(self, epoch_num):
        """
        Updates the training scheduler if any
        :param epoch_num: int,
        """
        gamma = 0.00495
        # if epoch_num > 0:
        #    self.beta += gamma
        if not self.has_reg_loss:
            if self.warm_up_epochs < epoch_num < 31:
                self.beta += gamma
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
            break
        print('LR: ', current_lr, ' Beta: ', self.beta)

    def step(self):
        """
        Perform the backward pass and step update for all optimizers
        :return:
        """
        self.optimizer.step()
        # self.scheduler.step()

    def compute_reg_loss(self, z, score):
        """
        Computes the regularization loss
        """
        if self.reg_type == 'rhy_complexity':
            attr_tensor = self.dataset.get_rhy_complexity(score)
        elif self.reg_type == 'int_entropy':
            attr_tensor = self.dataset.get_interval_entropy(score)
        elif self.reg_type == 'num_notes':
            attr_tensor = self.dataset.get_notes_density_in_measure(score)
        elif self.reg_type == 'note_range':
            attr_tensor = self.dataset.get_note_range_of_measure(score)
        elif self.reg_type == 'joint':
            attr1_tensor = self.dataset.get_rhy_complexity(score)
            attr2_tensor = self.dataset.get_notes_density_in_measure(score)
            x1 = z[:, self.reg_dim[0]]
            x2 = z[:, self.reg_dim[1]]
            reg_loss = self.reg_loss_sign(x1, attr1_tensor) + self.reg_loss_sign(x2, attr2_tensor)
            return reg_loss
        elif self.reg_type == 'joint_rhycomp_noterange':
            attr1_tensor = self.dataset.get_rhy_complexity(score)
            attr2_tensor = self.dataset.get_note_range_of_measure(score)
            x1 = z[:, self.reg_dim[0]]
            x2 = z[:, self.reg_dim[1]]
            reg_loss = self.reg_loss_sign(x1, attr1_tensor) + self.reg_loss_sign(x2, attr2_tensor)
            return reg_loss
        elif self.reg_type == 'four_metrics':
            attr1_tensor = self.dataset.get_rhy_complexity(score)
            attr2_tensor = self.dataset.get_note_range_of_measure(score)
            attr3_tensor = self.dataset.get_notes_density_in_measure(score)
            attr4_tensor = self.dataset.get_average_pitch_interval_of_measure(score)

            x1 = z[:, self.reg_dim[0]]
            x2 = z[:, self.reg_dim[1]]
            x3 = z[:, self.reg_dim[2]]
            x4 = z[:, self.reg_dim[3]]
            reg_loss = self.reg_loss_sign(x1, attr1_tensor) + self.reg_loss_sign(x2, attr2_tensor) + self.reg_loss_sign(x3, attr3_tensor) + self.reg_loss_sign(x4, attr4_tensor)
            return reg_loss

        else:
            raise ValueError('Invalid regularization attribute')
        x = z[:, self.reg_dim]
        reg_loss = self.reg_loss_sign(x=x, y=attr_tensor)
        return reg_loss

    def compute_kld_loss(self, z_dist, prior_dist):
        """
        :param z_dist: torch.nn.distributions object
        :param prior_dist: torch.nn.distributions
        :param beta:
        :return: kl divergence loss
        """
        kld = distributions.kl.kl_divergence(z_dist, prior_dist)
        kld = self.beta * kld.sum(1).mean()
        return kld

    @staticmethod
    def reg_loss_dist(x, y):
        """
        :param x: torch Variable,
        :param y: torch Variable,
        :return: scalar, loss
        """
        x_dist_mat = F.pdist(x.view(-1, 1))
        y_dist_mat = F.pdist(y.view(-1, 1))
        l1_loss = torch.nn.L1Loss()
        dist_loss = l1_loss(x_dist_mat, y_dist_mat)
        return dist_loss

    @staticmethod
    def reg_loss_sign(x, y):
        """
        :param x: torch Variable,
        :param y: torch Variable,
        :return: scalar, loss
        """
        # prepare data
        x = x.view(-1, 1).repeat(1, x.shape[0])
        x_diff_sign = (x - x.transpose(1, 0)).view(-1, 1)
        x_diff_sign = torch.tanh(x_diff_sign * 10)
        # prepare labels
        y = y.view(-1, 1).repeat(1, y.shape[0])
        y_diff_sign = torch.sign(y - y.transpose(1, 0)).view(-1, 1)
        # y_diff_sign[y_diff_sign == 1.] = 2.
        # y_diff_sign[y_diff_sign == 0.] = 1.
        # y_diff_sign[y_diff_sign == -1.] = 0.
        loss_fn = torch.nn.L1Loss()
        sign_loss = loss_fn(x_diff_sign, y_diff_sign)
        return sign_loss

    @staticmethod
    def reg_loss_corr(latent_vec, attribute, latent_dim=0):
        """
        :param latent_vec: torch Variable,
        :param attribute: torch Variable,
        :param latent_dim: int, latent dimension of interest
        """
        lv = latent_vec[:, latent_dim]
        lv = lv - torch.mean(lv)
        attr = attribute - torch.mean(attribute)
        lv_coeff = torch.sqrt(torch.sum(lv ** 2))
        attr_coeff = torch.sqrt(torch.sum(attr ** 2))
        reg_loss = 1. + torch.sum(lv * attr) / (lv_coeff * attr_coeff)
        return reg_loss

    @staticmethod
    def reg_coeff(attribute):
        """
        :param attribute: torch Variable,
        """
        attr = attribute - torch.mean(attribute)
        attr_coeff = torch.sqrt(torch.sum(attr ** 2))
        return attr_coeff

    @staticmethod
    def latent_loss(mu, sigma):
        """
        :param mu: torch Variable,
                    (batch_size, latent_space_dim)
        :param sigma: torch Variable,
                    (batch_size, latent_space_dim)
        :return: scalar, latent KL divergence loss
        """
        mean_sq = mu * mu
        sigma_sq = sigma * sigma
        ll = 0.5 * torch.mean(mean_sq + sigma_sq - torch.log(sigma_sq) - 1)
        return ll

    @staticmethod
    def compute_kernel(x, y, k):
        batch_size_x, dim_x = x.size()
        batch_size_y, dim_y = y.size()
        assert dim_x == dim_y

        xx = x.unsqueeze(1).expand(batch_size_x, batch_size_y, dim_x)
        yy = y.unsqueeze(0).expand(batch_size_x, batch_size_y, dim_y)
        distances = (xx - yy).pow(2).sum(2)
        return k(distances)

    @staticmethod
    def compute_mmd_loss(z_tilde, z_prior, coeff=10):
        """
        :param z_tilde:
        :param z_prior:
        :param coeff:
        :return:
        """
        # gaussian
        def gaussian(d, var=16.):
            return torch.exp(- d / var).sum(1).sum(0)

        # inverse multiquadratics
        def inverse_multiquadratics(d, var=16.):
            """
            :param d: (num_samples x, num_samples y)
            :param var:
            :return:
            """
            return (var / (var + d)).sum(1).sum(0)

        # k = inverse_multiquadratics
        k = gaussian
        batch_size = z_tilde.size(0)
        zp_ker = MeasureTrainer.compute_kernel(z_prior, z_prior, k)
        zt_ker = MeasureTrainer.compute_kernel(z_tilde, z_tilde, k)
        zp_zt_ker = MeasureTrainer.compute_kernel(z_prior, z_tilde, k)

        first_coeff = 1. / (batch_size * (batch_size - 1)) / 2 if batch_size > 1 else 1
        second_coeff = 2 / (batch_size * batch_size)
        mmd = coeff * (first_coeff * zp_ker
                       + first_coeff * zt_ker
                       - second_coeff * zp_zt_ker)
        return mmd

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

    if torch.cuda.is_available():
        model.cuda()
    trainer = MeasureTrainer(
        dataset=folk_dataset_train,
        model=model,
        lr=1e-4,
        has_reg_loss=reg_loss,
        reg_type=reg_type,
        reg_dim=reg_dim
    )
    trainer.train_model(
        batch_size=batch_size,
        num_epochs=num_epochs,
        plot=plot,
        log=log,
    )

if __name__ == "__main__":
    run()
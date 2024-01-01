
import torch

def get_train_fn(policy, optimizer, cfg):

    loss_func = torch.nn.MSELoss()
    ce_loss_func = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    #@torch.compile
    def train(ob_acs, returns, old_vpreds, old_neglogpacs, advs, rtm1, rew, gtp1, train_actor):

        policy.train()
        optimizer.zero_grad(True)

        neglogpac, vpred, entropy, rtm1_pred, rpred, gpred = policy.evaluate(**ob_acs)

        assert vpred.shape == neglogpac.shape == returns.shape == old_vpreds.shape == old_neglogpacs.shape == advs.shape
        returns = returns.detach()
        old_vpreds = old_vpreds.detach()
        old_neglogpacs = old_neglogpacs.detach()
        advs = advs.detach()
        rtm1 = rtm1.detach()
        rew = rew.detach()
        gtp1 = gtp1.detach()

        ratio = torch.exp(old_neglogpacs - neglogpac)
        surr1 = - ratio * advs
        surr2 = - torch.clamp(ratio, 1.0 - cfg.cliprange, 1.0 + cfg.cliprange) * advs

        actor_loss = torch.mean(torch.maximum(surr1, surr2))
        
        vpredclipped = old_vpreds + torch.clamp(vpred - old_vpreds, -cfg.cliprange, cfg.cliprange)
        vf_losses1 = loss_func(vpred, returns)
        vf_losses2 = loss_func(vpredclipped, returns)

        #critic_loss = .5 * torch.mean(torch.maximum(vf_losses1, vf_losses2)) * cfg.vf_coef
        critic_loss = .5 * torch.mean(vf_losses1) * cfg.vf_coef
        entropy_loss = - torch.mean(entropy) * cfg.ent_coef

        aux_loss1 = .5 * torch.mean(loss_func(rtm1_pred, rtm1)) * cfg.aux_coef
        aux_loss2 = .5 * torch.mean(loss_func(rpred, rew)) * cfg.aux_coef
        aux_loss3 = ce_loss_func(gpred.reshape(-1, gpred.shape[-1]), gtp1.reshape(-1)) * cfg.aux_coef

        # Final PG loss
        approxkl = .5 * torch.mean(torch.square(neglogpac - old_neglogpacs))
        clipfrac = torch.mean((torch.abs(ratio - 1.0) > cfg.cliprange).float())

        # Total loss
        loss = actor_loss * train_actor + entropy_loss + critic_loss + aux_loss1 + aux_loss2 + aux_loss3

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        optimizer.step()

        return loss, actor_loss, critic_loss, entropy_loss, aux_loss1, aux_loss2, aux_loss3, approxkl, clipfrac

    return train

def get_act_fn(policy):
    #@torch.compile
    def act(**state_info):
        with torch.no_grad():
            policy.eval()
            return policy.act(**state_info)
    return act
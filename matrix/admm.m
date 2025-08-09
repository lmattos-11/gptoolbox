function [X,Z,state] = admm(argmin_X,argmin_Z,A,B,c,state,varargin)
  % ADMM solver for convex problems of the form:
  %
  % min_X,Z f(X) + g(Z) subject to A*X + B*Z = c
  %
  % [X,Z,state] = admm(argmin_X,argmin_Z,A,B,c,state,varargin)
  %
  % Inputs:
  %   argmin_X  Function handle returning the optimizer of:
  %      argmin  f(X) + ρ/2‖ A*X + B*Z - c + U‖²
  %        X
  %      [X,data] = argmin_X(Z,U,rho,data)
  %      Inputs:
  %        Z  #Z by dim list of dual variables
  %        U  #c by dim list of scaled Lagrange multipliers
  %        rho  current penalty parameter
  %        data  empty [] on first call, or `data` ouput from previous call
  %      Outputs:
  %        X  #X by dim list of primary variables
  %        data  persistent callback data
  %   argmin_Z  Function handle returning the optimizer of:
  %      argmin  g(Z) + ρ/2‖ A*X + B*Z - c + U‖²
  %        Z
  %      [Z,data] = argmin_Z(X,U,rho,data)
  %      Inputs:
  %        X  #X by dim list of primary variables
  %        U  #c by dim list of scaled Lagrange multipliers
  %        rho  current penalty parameter
  %        data  empty [] on first call, or `data` ouput from previous call
  %      Outputs:
  %        Z  #Z by dim list of dual variables
  %        data  persistent callback data
  %   A  #c by #X constraint matrix coefficients corresponding to rows of X
  %   B  #c by #Z constraint matrix coefficients corresponding to rows of Z
  %   c  #c by dim list of constraint constants
  %   state  (see output)
  % Outputs
  %   X  #X by dim primary part of solution
  %   Z  #Z by dim dual part of solution
  %   state  struct containing persistent/reusable data
  %  
  %   

method = 'boyd';
max_iter = 2000;
callback = @(state) [];
tol_abs = 1e-8;
tol_rel = 1e-6;
check_interval = 10;
bmu = 5;
btao_inc = 2;
btao_dec = 2;
alpha = 1;
% Map of parameter names to variable names
params_to_variables = containers.Map( ...
{'Method','MaxIter' ,'Callback','TolAbs','TolRel','CheckInterval',...
        'PenaltyFactor','PenaltyIncrease','PenaltyDecrease','QuadratureConstant'}, ...
{'method','max_iter','callback','tol_abs','tol_rel','check_interval',...
        'bmu','btao_inc','btao_dec','alpha'});
v = 1;
while v <= numel(varargin)
param_name = varargin{v};
if isKey(params_to_variables,param_name)
  assert(v+1<=numel(varargin));
  v = v+1;
  % Trick: use feval on anonymous function to use assignin to this workspace
  feval(@()assignin('caller',params_to_variables(param_name),varargin{v}));
else
  error('Unsupported parameter: %s',varargin{v});
end
v=v+1;
end


  % Initial conditions
  if isempty(state) || ~isfield(state,'X')
    state.X = rand(size(A,2),size(c,2));
  end
  if isempty(state) || ~isfield(state,'Z')
    state.Z = rand(size(B,2),size(c,2));
  end
  if isempty(state) || ~isfield(state,'U')
    state.U = zeros(size(c,1),size(c,2));
  end
  if isempty(state) || ~isfield(state,'rho_prev')
    state.rho_prev = nan;
  end
  if isempty(state) || ~isfield(state,'rho')
    state.rho = 1;
  end
  if isempty(state) || ~isfield(state,'argmin_X_data')
    state.argmin_X_data = [];
  end
  if isempty(state) || ~isfield(state,'argmin_Z_data')
    state.argmin_Z_data = [];
  end

  optional_criterion = 1;
  cnorm = norm(c,'fro');
  state.X_initial = state.X;
  for iter = 1:max_iter
    state.iter = iter;
    state.X_prev = state.X;
    [state.X,state.argmin_X_data] = argmin_X(state.Z,state.U,state.rho,state.argmin_X_data);
    state.Z_prev = state.Z;
    [state.Z,state.argmin_Z_data] = argmin_Z(state.X,state.U,state.rho,state.argmin_Z_data);
    state.U_prev = state.U;
    callback(state);
    state.rho_prev = state.rho;
    switch method
        case 'boyd'
            state.U = state.U+A*state.X+B*state.Z-c;
            dual_residual = state.rho*norm(A'*B*(state.Z_prev - state.Z),'fro');
            residual = norm(A*state.X+B*state.Z-c,'fro');
            if mod(state.iter,check_interval) == 0
              if residual > bmu*dual_residual
                state.rho = btao_inc*state.rho;
                state.U = state.U/btao_inc;
              elseif dual_residual > bmu*residual
                state.rho = state.rho/btao_dec;
                state.U = state.U*btao_dec;
              end
            end
            k = size(c,2);
            eps_pri = sqrt(k*2)*tol_abs + tol_rel*max([norm(A*state.X,'fro'),norm(B*state.Z,'fro'),cnorm]);
            eps_dual = sqrt(k)*tol_abs +  tol_rel*state.rho*norm(A'*state.U,'fro');
        case 'stein'
            B = -state.argmin_Z_data.rotations;
            state.U = state.U+A*(alpha*state.X+(1-alpha)*state.X_initial)+times_3x3(B,state.Z);
            dual_residual_i = 0.5*state.rho .* norm_nx3(times_nx3(A',times_3x3(B,state.Z-state.Z_prev)));
            dual_residual = sqrt(sum(dual_residual_i.^2));
            residual_i = sum(reshape(( ( A*(alpha*state.X+(1-alpha)*state.X_initial)+times_3x3(B,state.Z) ).^2).', 9, [])).';
            residual = sqrt(sum(residual_i));
            if mod(state.iter,check_interval) == 0
                state.argmin_X_data.is_penalty_rescaled = any(state.rho(sqrt(residual_i) > bmu*dual_residual_i)) | any(state.rho(dual_residual_i > bmu*sqrt(residual_i)));
                state.rho(sqrt(residual_i) > bmu*dual_residual_i) = btao_inc*state.rho(sqrt(residual_i) > bmu*dual_residual_i);
                state.U(repelem(sqrt(residual_i) > bmu*dual_residual_i,3),:) = state.U(repelem(sqrt(residual_i) > bmu*dual_residual_i,3),:)/btao_inc;
                state.rho(dual_residual_i > bmu*sqrt(residual_i)) = state.rho(dual_residual_i > bmu*sqrt(residual_i))/btao_dec;
                state.U(repelem(dual_residual_i > bmu*sqrt(residual_i),3),:) = state.U(repelem(dual_residual_i > bmu*sqrt(residual_i),3),:)*btao_dec;        
            end
            k = size(c,1);
            eps_pri = sqrt(k)*tol_abs + tol_rel*max([sum(sqrt(sum(reshape(sum((A*(alpha*state.X+(1-alpha)*state.X_initial)).^2,2)',3,[]),1))),...
                                                        sum(sqrt(sum(reshape(sum((state.Z).^2,2)',3,[]),1)))]);
            eps_dual = sqrt(k)*tol_abs + tol_rel*sum(norm_nx3(times_nx3(A',state.U)));
            optional_criterion = min(eig_3x3(state.Z)) >= 0;
            %fprintf('Residual: %s Tol: %s Dual residual: %s Tol: %s\n',full(residual),full(eps_pri), full(dual_residual), full(eps_dual))
    end
    if residual < eps_pri && dual_residual < eps_dual && optional_criterion
      break;
    end
  end
  X = state.X;
  Z = state.Z;
end
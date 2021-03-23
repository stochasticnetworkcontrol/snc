snc.environments.closed\_loop\_crw.ClosedLoopCRW
================================================

.. currentmodule:: snc.environments.closed_loop_crw

.. autoclass:: ClosedLoopCRW
   :members:
   :show-inheritance:
   :inherited-members:
   :special-members: __call__, __add__, __mul__

   
   
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
   
      ~ClosedLoopCRW.are_demand_ids_unique
      ~ClosedLoopCRW.assert_remains_closed_network
      ~ClosedLoopCRW.close
      ~ClosedLoopCRW.ensure_jobs_conservation
      ~ClosedLoopCRW.fill_in_transit_to_suppliers
      ~ClosedLoopCRW.fill_supply_buffers
      ~ClosedLoopCRW.get_activity_supply_resource_association
      ~ClosedLoopCRW.get_num_items_in_transit_to_suppliers
      ~ClosedLoopCRW.get_num_items_state_without_demand
      ~ClosedLoopCRW.get_num_items_supply_buff
      ~ClosedLoopCRW.get_satisfied_demand
      ~ClosedLoopCRW.get_supply_activity_to_buffer_association
      ~ClosedLoopCRW.get_supply_and_demand_ids
      ~ClosedLoopCRW.initialise_supply_buffers
      ~ClosedLoopCRW.is_demand_to_supplier_routes_consistent_with_job_generator
      ~ClosedLoopCRW.is_supply_ids_consistent_with_job_generator
      ~ClosedLoopCRW.is_surplus_buffers_consistent_with_job_generator
      ~ClosedLoopCRW.render
      ~ClosedLoopCRW.reset
      ~ClosedLoopCRW.reset_with_random_state
      ~ClosedLoopCRW.seed
      ~ClosedLoopCRW.step
      ~ClosedLoopCRW.sum_supplier_outbound
      ~ClosedLoopCRW.to_serializable
      ~ClosedLoopCRW.truncate_routing_matrix_supplier
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~ClosedLoopCRW.action_space
      ~ClosedLoopCRW.capacity
      ~ClosedLoopCRW.constituency_matrix
      ~ClosedLoopCRW.cost_per_buffer
      ~ClosedLoopCRW.energy_conservation_flag
      ~ClosedLoopCRW.ind_surplus_buffers
      ~ClosedLoopCRW.index_phys_resources
      ~ClosedLoopCRW.is_at_final_state
      ~ClosedLoopCRW.is_at_initial_state
      ~ClosedLoopCRW.job_generator
      ~ClosedLoopCRW.list_boundary_constraint_matrices
      ~ClosedLoopCRW.max_episode_length
      ~ClosedLoopCRW.metadata
      ~ClosedLoopCRW.model_type
      ~ClosedLoopCRW.num_activities
      ~ClosedLoopCRW.num_buffers
      ~ClosedLoopCRW.num_resources
      ~ClosedLoopCRW.observation_space
      ~ClosedLoopCRW.physical_constituency_matrix
      ~ClosedLoopCRW.reward_range
      ~ClosedLoopCRW.spec
      ~ClosedLoopCRW.state_initialiser
      ~ClosedLoopCRW.t
      ~ClosedLoopCRW.unwrapped
   
   
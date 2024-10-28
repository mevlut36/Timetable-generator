from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_timetables():
    num_classes = int(input("Entrez le nombre de classes : "))

    subjects = ["Français", "Anglais", "Math", "Histoire Géo", "Philosophie", "EPS"]

    teachers = {}
    for subject in subjects:
        num_teachers = int(input(f"Entrez le nombre de professeurs disponibles pour {subject} : "))
        teachers[subject] = num_teachers

    teacher_list = []
    teacher_number = {}
    teacher_counter = 1
    for subject in subjects:
        for t in range(teachers[subject]):
            teacher = (subject, t)
            teacher_list.append(teacher)
            teacher_number[teacher] = teacher_counter
            teacher_counter += 1

    lessons_per_week = {}
    for subject in subjects:
        num_lessons = int(input(f"Entrez le nombre de cours par semaine pour {subject} : "))
        lessons_per_week[subject] = num_lessons

    days = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi"]

    morning_slots = list(range(8, 13))   # 8h - 12h
    afternoon_slots = list(range(14, 18))  # 14h - 17h
    time_slots = morning_slots + afternoon_slots

    block_start_times = [8, 14]

    model = cp_model.CpModel()

    schedule = {}
    teaches_vars = {}
    for cls in range(num_classes):
        for day in days:
            for hour in time_slots:
                for subject in subjects:
                    schedule[(cls, day, hour, subject)] = model.NewBoolVar(f'schedule_c{cls}_d{day}_h{hour}_s{subject}')
                    for teacher_id in range(teachers[subject]):
                        teacher = (subject, teacher_id)
                        teaches = model.NewBoolVar(f'teaches_{teacher}_{cls}_{day}_{hour}_{subject}')
                        teaches_vars[(cls, day, hour, subject, teacher)] = teaches

    teacher_blocks = {}
    for subject, teacher_id in teacher_list:
        for day in days:
            for start_hour in block_start_times:
                teacher_blocks[(subject, teacher_id, day, start_hour)] = model.NewBoolVar(f'teacherBlock_{subject}_{teacher_id}_{day}_{start_hour}')

    for subject, teacher_id in teacher_list:
        for day in days:
            model.Add(sum(teacher_blocks[(subject, teacher_id, day, start_hour)] for start_hour in block_start_times) <= 1)

    for cls in range(num_classes):
        for day in days:
            for hour in time_slots:
                for subject in subjects:
                    teacher_vars = []
                    for teacher_id in range(teachers[subject]):
                        teacher = (subject, teacher_id)
                        in_block = []
                        for start_hour in block_start_times:
                            block_hours = [start_hour, start_hour + 1, start_hour + 2]
                            if start_hour + 2 in time_slots:
                                block_hours = [start_hour, start_hour + 1, start_hour + 2]
                            else:
                                block_hours = [start_hour, start_hour + 1]
                            if hour in block_hours:
                                in_block.append(teacher_blocks[(subject, teacher_id, day, start_hour)])
                        if in_block:
                            is_in_block = model.NewBoolVar(f'isInBlock_{teacher}_{day}_{hour}')
                            model.AddMaxEquality(is_in_block, in_block)
                            teaches = teaches_vars[(cls, day, hour, subject, teacher)]
                            model.Add(teaches <= schedule[(cls, day, hour, subject)])
                            model.Add(teaches <= is_in_block)
                            teacher_vars.append(teaches)
                        else:
                            model.Add(teaches_vars[(cls, day, hour, subject, teacher)] == 0)
                    model.Add(schedule[(cls, day, hour, subject)] == sum(teaches_vars[(cls, day, hour, subject, teacher)] for teacher in [(subject, t) for t in range(teachers[subject])]))
                    model.Add(sum(teaches_vars[(cls, day, hour, subject, teacher)] for teacher in [(subject, t) for t in range(teachers[subject])]) <= 1)

    for cls in range(num_classes):
        for day in days:
            for hour in time_slots:
                model.Add(sum(schedule[(cls, day, hour, subject)] for subject in subjects) <= 1)

    for subject, teacher_id in teacher_list:
        for day in days:
            for hour in time_slots:
                teaches = []
                for cls in range(num_classes):
                    key = (cls, day, hour, subject, (subject, teacher_id))
                    if key in teaches_vars:
                        teaches.append(teaches_vars[key])
                model.Add(sum(teaches) <= 1)

    for cls in range(num_classes):
        for subject in subjects:
            total_lessons = sum(schedule[(cls, day, hour, subject)]
                                for day in days
                                for hour in time_slots)
            model.Add(total_lessons == lessons_per_week[subject])

    for cls in range(num_classes):
        for day in days:
            lessons_per_day = sum(schedule[(cls, day, hour, subject)]
                                  for hour in time_slots
                                  for subject in subjects)
            model.Add(lessons_per_day >= 1)
            model.Add(lessons_per_day <= 6)

    for cls in range(num_classes):
        for subject in subjects:
            days_with_subject = []
            for day in days:
                has_subject_today = model.NewBoolVar(f'Class{cls}_{subject}_{day}_Scheduled')
                model.AddMaxEquality(has_subject_today,
                                     [schedule[(cls, day, hour, subject)] for hour in time_slots])
                days_with_subject.append(has_subject_today)
            min_days = min(lessons_per_week[subject], 2)
            model.Add(sum(days_with_subject) >= min_days)

    has_course = {}
    for cls in range(num_classes):
        for day in days:
            for hour in time_slots:
                var = model.NewBoolVar(f'HasCourse_Class{cls}_{day}_{hour}')
                model.AddMaxEquality(var, [schedule[(cls, day, hour, subject)] for subject in subjects])
                has_course[(cls, day, hour)] = var

    gaps = []
    for cls in range(num_classes):
        for day in days:
            for idx in range(len(time_slots)-1):
                hour = time_slots[idx]
                next_hour = time_slots[idx+1]
                if hour == 12:
                    continue
                gap = model.NewBoolVar(f'Gap_Class{cls}_{day}_{hour}')
                model.AddBoolAnd([
                    has_course[(cls, day, hour)],
                    has_course[(cls, day, next_hour)].Not()
                ]).OnlyEnforceIf(gap)
                model.AddImplication(gap, has_course[(cls, day, hour)])
                gaps.append(gap)

    long_gaps = []
    for cls in range(num_classes):
        for day in days:
            for idx in range(len(time_slots)-2):
                hour = time_slots[idx]
                next_hour = time_slots[idx+1]
                next_next_hour = time_slots[idx+2]
                if hour == 11 or hour == 12:
                    continue
                long_gap = model.NewBoolVar(f'LongGap_Class{cls}_{day}_{hour}')
                model.AddBoolAnd([
                    has_course[(cls, day, hour)],
                    has_course[(cls, day, next_hour)].Not(),
                    has_course[(cls, day, next_next_hour)]
                ]).OnlyEnforceIf(long_gap)
                model.AddImplication(long_gap, has_course[(cls, day, hour)])
                long_gaps.append(long_gap)

    weight_gaps = 1
    weight_long_gaps = 10
    model.Minimize(
        weight_gaps * sum(gaps) +
        weight_long_gaps * sum(long_gaps)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 600.0
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Emplois du temps générés avec succès.")

        student_output_folder = 'emplois_du_temps_eleve'
        teacher_output_folder = 'emplois_du_temps_prof'
        if not os.path.exists(student_output_folder):
            os.makedirs(student_output_folder)
        if not os.path.exists(teacher_output_folder):
            os.makedirs(teacher_output_folder)

        for cls in range(num_classes):
            timetable = np.empty((len(time_slots), len(days)), dtype=object)
            for hour_idx, hour in enumerate(time_slots):
                for day_idx, day in enumerate(days):
                    subject_found = False
                    for subject in subjects:
                        if solver.Value(schedule[(cls, day, hour, subject)]):
                            assigned_teacher = None
                            for teacher_id in range(teachers[subject]):
                                teacher = (subject, teacher_id)
                                key = (cls, day, hour, subject, teacher)
                                if key in teaches_vars and solver.Value(teaches_vars[key]):
                                    assigned_teacher = f"M. {teacher_number[teacher]}"
                                    break
                            if assigned_teacher:
                                timetable[hour_idx][day_idx] = f"{subject}\n{assigned_teacher}"
                            else:
                                timetable[hour_idx][day_idx] = subject
                            subject_found = True
                            break
                    if not subject_found:
                        timetable[hour_idx][day_idx] = ''
            fig_height = len(time_slots) * 0.5
            fig, ax = plt.subplots(figsize=(10, fig_height))
            ax.axis('off')
            table_data = [ [timetable[i][j] for j in range(len(days))] for i in range(len(time_slots)) ]
            table = ax.table(cellText=table_data,
                             rowLabels=[f'{hour}h' for hour in time_slots],
                             colLabels=days,
                             cellLoc='center',
                             loc='center')

            table.auto_set_font_size(False)
            table.set_fontsize(10)

            cell_height = 1 / (len(time_slots) + 1)
            for (i, j), cell in table.get_celld().items():
                cell.set_height(cell_height)

            subject_colors = {
                'Français': '#FFC0CB',
                'Anglais': '#ADD8E6',
                'Math': '#90EE90',
                'Histoire Géo': '#FFFFE0',
                'Philosophie': '#D3D3D3', 
                'EPS': '#FFA07A',
                '': '#FFFFFF'
            }
            for (i, j), cell in table.get_celld().items():
                if (i == 0) or (j == -1):
                    cell.set_text_props(weight='bold', color='black')
                    cell.set_facecolor('#D3D3D3')
                else:
                    content = table_data[i-1][j]
                    if content:
                        subject_name = content.split('\n')[0]
                        cell.set_facecolor(subject_colors.get(subject_name, '#FFFFFF'))
                    else:
                        cell.set_facecolor('#FFFFFF')
            plt.title(f'Emploi du temps de la Classe {cls+1}', fontsize=16)
            plt.savefig(os.path.join(student_output_folder, f'Classe_{cls+1}.png'))
            plt.close()

        for teacher in teacher_list:
            timetable = np.empty((len(time_slots), len(days)), dtype=object)
            for hour_idx, hour in enumerate(time_slots):
                for day_idx, day in enumerate(days):
                    class_found = False
                    for cls in range(num_classes):
                        subject = teacher[0]
                        key = (cls, day, hour, subject, teacher)
                        if key in teaches_vars and solver.Value(teaches_vars[key]):
                            timetable[hour_idx][day_idx] = f"{subject}\nClasse {cls+1}"
                            class_found = True
                            break
                    if class_found:
                        break
                    else:
                        timetable[hour_idx][day_idx] = ''
            fig_height = len(time_slots) * 0.5
            fig, ax = plt.subplots(figsize=(10, fig_height))
            ax.axis('off')
            table_data = [ [timetable[i][j] for j in range(len(days))] for i in range(len(time_slots)) ]
            table = ax.table(cellText=table_data,
                             rowLabels=[f'{hour}h' for hour in time_slots],
                             colLabels=days,
                             cellLoc='center',
                             loc='center')

            table.auto_set_font_size(False)
            table.set_fontsize(10)

            cell_height = 1 / (len(time_slots) + 1)
            for (i, j), cell in table.get_celld().items():
                cell.set_height(cell_height)

            subject_colors = {
                'Français': '#FFC0CB',
                'Anglais': '#ADD8E6',
                'Math': '#90EE90',
                'Histoire Géo': '#FFFFE0', 
                'Philosophie': '#D3D3D3',
                'EPS': '#FFA07A',
                '': '#FFFFFF'
            }
            for (i, j), cell in table.get_celld().items():
                if (i == 0) or (j == -1):
                    cell.set_text_props(weight='bold', color='black')
                    cell.set_facecolor('#D3D3D3')
                else:
                    content = table_data[i-1][j]
                    if content:
                        subject_name = content.split('\n')[0]
                        cell.set_facecolor(subject_colors.get(subject_name, '#FFFFFF'))
                    else:
                        cell.set_facecolor('#FFFFFF')
            teacher_name = f"M. {teacher_number[teacher]}"
            plt.title(f'Emploi du temps de {teacher_name}', fontsize=16)
            plt.savefig(os.path.join(teacher_output_folder, f'Prof_{teacher_name}.png'))
            plt.close()
    else:
        print("Aucune solution trouvée. Veuillez vérifier les paramètres et essayer à nouveau.")

if __name__ == "__main__":
    generate_timetables()
